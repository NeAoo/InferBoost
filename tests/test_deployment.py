import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pipeline import deployment
from pipeline.cli import retest, select, serve
from pipeline.runtime import build_command


class DeploymentContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / 'runs' / 'r1'
        self.run.mkdir(parents=True)
        self.profile = {'id': 'p1', 'attention': 'fa2', 'graph': True, 'mtp': 0, 'batched_tokens': 8192}
        self.config = {
            'schema_version': 1,
            'runtime': {'python': sys.executable, 'gpu': 0,
                        'environment': {'PRIVATE_TEST_SECRET': 'not-for-export', 'ACCEL60_DRAFT_CONFIG_FIX': '1'}},
            'quantization': {'mode': 'fresh', 'method': 'awq4', 'source': '/models/bf16'},
            'audit': {}, 'server': {'port': 18450, 'max_model_len': 32768, 'max_num_seqs': 4},
            'benchmark': {'dataset': '/data/public100.json', 'concurrencies': [1], 'cache_modes': ['cold', 'warm']},
            'profiles': [self.profile],
        }
        self.model = self.root / 'models' / 'awq4'
        self.model.mkdir(parents=True)
        (self.model / 'config.json').write_text('{}')
        (self.run / 'model').mkdir()
        (self.run / 'model/config.json').write_text('{}')
        self.manifest = {'source_model': str(self.model), 'load_view': str(self.run / 'model'),
                         'identity': {'shards': {'model.safetensors': {'sha256': 'fixture-sha256'}}}}
        self.write('config.json', self.config)
        self.write('model_manifest.json', self.manifest)
        self.write('doctor.json', {'gpu': 'A800', 'capability': [8, 0], 'vllm': '0.33.0'})
        self.write('profiles/p1/status.json', {'status': 'completed'})
        self.write('profiles/p1/launch.json', {
            'profile': self.profile,
            'command': build_command(self.config, self.profile, self.manifest['load_view'], 18450)})
        self.write('profiles/p1/loading_audit.json', {'passed': True, 'versions': {'vllm': '0.33.0', 'torch': 'fixture'}})

    def write(self, name, value):
        path = self.run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_export_contains_recipe_and_fixed_start_profile_without_secrets(self):
        directory = deployment.export(self.root, self.run, 'p1')
        recipe = json.loads((directory / 'deployment.json').read_text())
        self.assertEqual(recipe['model'], self.manifest)
        self.assertEqual(recipe['versions']['vllm'], '0.33.0')
        self.assertNotIn('PRIVATE_TEST_SECRET', recipe['environment'])
        self.assertEqual(recipe['environment']['VLLM_SERVER_DEV_MODE'], '0')
        self.assertEqual(recipe['environment']['ACCEL60_DRAFT_CONFIG_FIX'], '1')
        self.assertEqual(recipe['runtime_build']['python_executable'], sys.executable)
        self.assertFalse(recipe['weights_included'])
        self.assertIsNone(recipe['image'])
        # Exercise argument forwarding without starting vLLM.
        quickstart = self.root / 'quickstart'
        quickstart.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGUMENT_LOG"\n')
        quickstart.chmod(0o755)
        output = self.root / 'arguments.txt'
        subprocess.run([str(directory / 'start.sh'), '--smoke-test', '--port', '9001'], check=True,
                       env={**os.environ, 'ARGUMENT_LOG': str(output)})
        self.assertEqual(output.read_text().splitlines(),
                         ['serve', '--run-id', 'r1', '--profile', 'p1', '--smoke-test', '--port', '9001'])

    def test_modified_untested_profile_cannot_be_exported_or_selected(self):
        self.config['profiles'][0]['batched_tokens'] = 4096
        self.write('config.json', self.config)
        with self.assertRaises(ValueError):
            select(self.root, self.run, 'p1')
        self.assertFalse((self.run / 'selected.json').exists())

    def test_retest_reuses_weights_and_keeps_the_baseline(self):
        before = (self.run / 'config.json').read_bytes()
        retest(self.root, self.run, 'p1', [4096, 8192], 'budget-test', prepare_only=True)
        config = json.loads((self.root / 'configs/budget-test.local.json').read_text())
        self.assertEqual(config['quantization']['mode'], 'reuse')
        self.assertEqual(config['quantization']['model'], str(self.model))
        self.assertEqual([profile['batched_tokens'] for profile in config['profiles']], [8192, 4096])
        self.assertEqual(config['benchmark']['concurrencies'], [1])
        self.assertEqual(config['benchmark']['cache_modes'], ['cold', 'warm'])
        self.assertEqual((self.run / 'config.json').read_bytes(), before)
        self.assertFalse((self.root / 'runs/budget-test').exists())

    def test_invalid_budget_does_not_create_a_retest(self):
        with self.assertRaises(ValueError):
            retest(self.root, self.run, 'p1', [1], 'bad-budget', prepare_only=True)
        self.assertFalse((self.root / 'configs/bad-budget.local.json').exists())

    def test_edited_selection_is_rejected_before_starting_a_server(self):
        self.write('selected.json', {'profile': {**self.profile, 'batched_tokens': 4096}})
        with self.assertRaises(ValueError):
            serve(self.root, self.run, smoke=True)
        self.write('selected.json', {'profile': self.profile})
        self.config['server']['max_model_len'] = 16384
        self.write('config.json', self.config)
        with self.assertRaises(ValueError):
            serve(self.root, self.run, smoke=True)


if __name__ == '__main__':
    unittest.main()
