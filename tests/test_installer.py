"""Host installation contracts with a stateful Docker protocol fixture, no GPU claims."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from installer import runtime

IMAGE_ID = 'sha256:' + 'a' * 64
FAKE_DOCKER = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args=sys.argv[1:]
root=Path(os.environ['INFERBOOST_TEST_DOCKER_STATE'])
with (root/'calls.jsonl').open('a') as stream:stream.write(json.dumps(args)+'\n')
if args[:1]==['--context']:args=args[2:]
if args[:1]==['info']:print(json.dumps({'OSType':'linux','Architecture':'x86_64'}))
elif args[:2]==['context','inspect']:print(json.dumps([{'Endpoints':{'docker':{'Host':'unix:///var/run/docker.sock'}}}]))
elif args[:2]==['image','inspect']:print(json.dumps([{'Id':'sha256:'+'a'*64,'Os':'linux','Architecture':'amd64','RepoDigests':[], 'Config':{'Labels':{'org.opencontainers.image.title':'InferBoost','org.opencontainers.image.version':'__IMAGE_VERSION__'}}}]))
elif args[0]=='run':
 if os.environ.get('INFERBOOST_TEST_GPU_FAILURE'):sys.exit(42)
 print('{"passed":true,"writable":true,"fixture":true}')
elif args[:2]==['container','inspect']:
 path=root/(args[-1]+'.json')
 if not path.exists():print('Error: No such container',file=sys.stderr);sys.exit(1)
 print('['+path.read_text()+']')
elif args[0]=='create':
 name=args[args.index('--name')+1]
 labels=dict(args[i+1].split('=',1) for i in range(len(args)) if args[i]=='--label')
 (root/(name+'.json')).write_text(json.dumps({'Image':'sha256:'+'a'*64,'Config':{'Labels':labels},'State':{'Running':False}}))
 print(name)
elif args[0] in ('start','stop'):
 path=root/(args[-1]+'.json');state=json.loads(path.read_text());state['State']['Running']=args[0]=='start';path.write_text(json.dumps(state));print(args[-1])
else:print('unexpected '+str(args),file=sys.stderr);sys.exit(5)
'''


class HostInstallerContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.models = self.root / 'models with spaces'
        self.models.mkdir()
        self.workspace = self.root / 'persistent work'
        self.bin = self.root / 'user bin'
        self.config = self.root / 'config'
        docker = self.root / 'docker'
        docker.write_text(FAKE_DOCKER.replace('__IMAGE_VERSION__', runtime.IMAGE_VERSION))
        docker.chmod(0o755)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ, {'INFERBOOST_TEST_DOCKER_STATE': str(self.root)}))
        stack.enter_context(patch.object(runtime, 'validate_host'))
        stack.enter_context(patch.object(runtime, 'config_home', return_value=self.config))
        stack.enter_context(patch.object(runtime.shutil, 'which', return_value=str(docker)))
        stack.enter_context(patch.object(runtime.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(200*1024**3, 0, 200*1024**3)))
        self.arguments = ['--image', 'inferboost:0.5.0-cu130', '--image-id', IMAGE_ID,
            '--models', str(self.models), '--workspace', str(self.workspace), '--bin-dir', str(self.bin)]

    def test_repeated_install_pins_identity_and_reuses_background_container(self):
        runtime.install(self.arguments)
        first = (self.config / 'config.json').read_bytes()
        runtime.install(self.arguments)
        self.assertEqual(first, (self.config / 'config.json').read_bytes())
        with patch.object(runtime.os, 'execv') as execute:
            runtime.launch([])
            runtime.launch([])
        calls = [json.loads(line) for line in (self.root / 'calls.jsonl').read_text().splitlines()]
        self.assertEqual(sum(args[0] == 'create' for args in calls), 1)
        creation = next(args for args in calls if args[0] == 'create')
        self.assertNotIn('--rm', creation)
        self.assertIn('type=bind,source=' + str(self.models.resolve()) + ',target=/models,readonly', creation)
        self.assertEqual(execute.call_args.args[1][-1], 'quickstart')
        runtime.launch(['stop'])
        runtime.launch(['status'])
        self.assertTrue((self.workspace).is_dir())

    def test_installer_targets_the_application_release_and_rejects_other_versions(self):
        from pipeline.product import VERSION as application_version
        self.assertEqual(runtime.IMAGE_VERSION, application_version)
        metadata = {'Id': IMAGE_ID, 'Os': 'linux', 'Architecture': 'amd64', 'Config': {'Labels': {
            'org.opencontainers.image.title': 'InferBoost', 'org.opencontainers.image.version': application_version}}}
        with patch.object(runtime, 'capture', return_value=json.dumps([metadata])):
            self.assertEqual(runtime.validate_image({}, 'example'), IMAGE_ID)
        metadata['Config']['Labels']['org.opencontainers.image.version'] = 'other-release'
        with patch.object(runtime, 'capture', return_value=json.dumps([metadata])):
            with self.assertRaisesRegex(ValueError, '实际镜像版本'):
                runtime.validate_image({}, 'example')

    def test_wrong_identity_fails_before_config_is_written(self):
        bad = self.arguments.copy();bad[bad.index('--image-id') + 1] = 'sha256:' + 'b'*64
        with self.assertRaisesRegex(ValueError, 'ID'):
            runtime.install(bad)
        self.assertFalse((self.config / 'config.json').exists())

    def test_cuda_failure_preserves_previous_installation(self):
        runtime.install(self.arguments)
        before = (self.config / 'config.json').read_bytes()
        with patch.dict(os.environ, {'INFERBOOST_TEST_GPU_FAILURE': '1'}):
            with self.assertRaises(subprocess.CalledProcessError):
                runtime.install(self.arguments)
        self.assertEqual(before, (self.config / 'config.json').read_bytes())

    def test_unrelated_launcher_is_not_overwritten(self):
        self.bin.mkdir();(self.bin/'inferboost').write_text('other program')
        with self.assertRaisesRegex(ValueError, '未覆盖'):
            runtime.install(self.arguments)
        self.assertEqual((self.bin/'inferboost').read_text(), 'other program')
        self.assertFalse((self.config/'config.json').exists())

    def test_dev_mount_and_container_are_distinct(self):
        runtime.install(self.arguments)
        config = json.loads((self.config/'config.json').read_text())
        config['port'] = 19001
        name, args = runtime.container_options(config, '/code with spaces')
        self.assertNotEqual(name, runtime.container_name(config))
        self.assertIn('type=bind,source=/code with spaces,target=/src', args)
        self.assertIn('PIPELINE_PYTHON=/opt/inferboost-ui/bin/python', args)
        self.assertNotIn('--publish', args)

    def test_arguments_are_not_shell_executed(self):
        value = self.root / 'models $(touch SHOULD_NOT_EXIST)'
        value.mkdir()
        arguments = self.arguments.copy();arguments[arguments.index('--models')+1] = str(value)
        runtime.install(arguments)
        config = json.loads((self.config/'config.json').read_text())
        self.assertEqual(config['models'], str(value.resolve()))
        self.assertFalse((Path.cwd()/'SHOULD_NOT_EXIST').exists())

    def test_existing_data_is_not_chowned_when_container_cannot_write(self):
        self.workspace.mkdir()
        (self.workspace/'keep.txt').write_text('existing result')
        result = json.dumps({'writable': False, 'error': 'permission denied',
                             'uid_map': '0 165536 65536', 'gid_map': '0 165536 65536'})
        with patch.object(runtime, 'capture', return_value=result), patch.object(runtime.os, 'chown') as change:
            with self.assertRaisesRegex(ValueError, '已有数据目录'):
                runtime.verify_workspace_access({'workspace': str(self.workspace), 'image_id': IMAGE_ID})
            change.assert_not_called()

    def test_root_empty_workspace_user_namespace_mapping_is_checked(self):
        self.workspace.mkdir()
        results = [json.dumps({'writable': False, 'uid_map': '0 165536 65536',
                              'gid_map': '0 165536 65536'}), json.dumps({'writable': True})]
        with patch.object(runtime, 'capture', side_effect=results), patch.object(runtime.os, 'getuid', return_value=0), \
                patch.object(runtime.os, 'getgid', return_value=0), patch.object(runtime.os, 'chown') as change:
            runtime.verify_workspace_access({'workspace': str(self.workspace), 'image_id': IMAGE_ID})
            change.assert_called_once_with(self.workspace, 165536, 165536)


if __name__ == '__main__':
    unittest.main()
