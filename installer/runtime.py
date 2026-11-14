#!/usr/bin/env python3
"""InferBoost host installer/launcher. Standard library only; never evaluates shell input."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

VERSION = '0.5.0'
IMAGE_VERSION = '0.5.0-r11'
WORKSPACE_NAME = 'inferboost-v050'


def config_home():
    return Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'inferboost'


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def config_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'launcher.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def command(config, *arguments):
    docker = [config.get('docker', 'docker')]
    if config.get('context'):
        docker += ['--context', config['context']]
    return docker + list(arguments)


def capture(config, *arguments, timeout=30):
    result = subprocess.run(command(config, *arguments), capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'Docker命令失败')
    return result.stdout.strip()


def directory(value, create=False):
    path = Path(value).expanduser().resolve()
    if any(character in str(path) for character in ('\n', '\r', ',', ':')):
        raise ValueError('Docker挂载路径不能包含换行、逗号或冒号；路径中的空格可以使用。')
    if path == Path('/'):
        raise ValueError('请指定专门的模型或工作目录，不要挂载整个根目录。')
    if create:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path) as stream:
            stream.write(b'write-check')
    if not path.is_dir():
        raise ValueError('目录不存在：' + str(path))
    return str(path)


def validate_host():
    if sys.version_info < (3, 10):
        raise ValueError('主机安装器需要Python 3.10+；不必在主机安装Torch或CUDA工具链。')
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'amd64'):
        raise ValueError('推理安装器要求Linux amd64与NVIDIA GPU；macOS可构建/上传镜像，不能执行本GPU流程。')


def validate_image(config, reference, expected_id=None, expected_digest=None):
    info = json.loads(capture(config, 'image', 'inspect', reference))[0]
    if info.get('Os') != 'linux' or info.get('Architecture') != 'amd64':
        raise ValueError('镜像必须为linux/amd64')
    if expected_id and info['Id'] != expected_id:
        raise ValueError('本地镜像ID与指定发布身份不一致')
    if expected_digest:
        digests = [value.rsplit('@', 1)[-1] for value in info.get('RepoDigests', [])]
        if expected_digest not in digests:
            raise ValueError('仓库digest与指定发布身份不一致')
    labels = info.get('Config', {}).get('Labels', {})
    if labels.get('org.opencontainers.image.title') != 'InferBoost' or labels.get('org.opencontainers.image.version') != IMAGE_VERSION:
        raise ValueError('安装器需要 InferBoost ' + IMAGE_VERSION + '，实际镜像版本为 ' + str(labels.get('org.opencontainers.image.version')))
    return info['Id']


def container_name(config, source=None):
    identity = dict(config, source=source)
    if source:
        identity['port'] = None
    identity.pop('container', None)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    return 'inferboost-' + ('dev-' if source else '') + digest


def verify_workspace_access(config):
    """Probe the actual bind mount, including Docker user-namespace mappings."""
    probe = '''import json,os,tempfile
from pathlib import Path
result={'uid_map':Path('/proc/self/uid_map').read_text(),'gid_map':Path('/proc/self/gid_map').read_text()}
try:
 with tempfile.TemporaryFile(dir='/workspace') as stream: stream.write(b'InferBoost write check')
 result['writable']=True
except OSError as error:
 result.update(writable=False,error=str(error))
print(json.dumps(result))
'''
    def inspect_access():
        return json.loads(capture(config, 'run', '--rm', '--user', f'{os.getuid()}:{os.getgid()}',
            '--mount', f"type=bind,source={config['workspace']},target=/workspace",
            '--entrypoint', '/opt/venv/bin/python', config['image_id'], '-c', probe))
    result = inspect_access()
    if result.get('writable'):
        return
    workspace = Path(config['workspace'])
    if os.getuid() == 0 and not any(workspace.iterdir()):
        def mapped_id(value, mapping):
            for line in mapping.splitlines():
                inside_id, outside_id, count = map(int, line.split())
                if inside_id <= value < inside_id + count:
                    return outside_id + value - inside_id
            raise ValueError('Docker用户命名空间中没有对应UID/GID')
        uid = mapped_id(os.getuid(), result['uid_map'])
        gid = mapped_id(os.getgid(), result['gid_map'])
        if (uid, gid) != (os.getuid(), os.getgid()):
            previous = workspace.stat()
            print(f'Docker启用了用户映射；为尚无文件的工作目录设置容器所有者 {uid}:{gid}。', flush=True)
            os.chown(workspace, uid, gid)
            success = False
            try:
                success = bool(inspect_access().get('writable'))
                if success:
                    return
            finally:
                if not success:
                    os.chown(workspace, previous.st_uid, previous.st_gid)
    raise ValueError('容器不能写入工作卷；请为Docker映射后的用户配置目录权限，或换一个可写目录。'
                     '已有数据目录不会被自动改权限。详情：' + result.get('error', 'write probe failed'))


def container_options(config, source=None):
    name = container_name(config, source)
    args = ['create', '--name', name, '--label', 'io.inferboost.managed=true',
            '--label', 'io.inferboost.configuration=' + name, '--init',
            '--gpus', 'all' if config['gpus'] == 'all' else '"device=' + config['gpus'] + '"',
            '--shm-size', config['shm_size'], '--user', f"{os.getuid()}:{os.getgid()}",
            '--mount', f"type=bind,source={config['models']},target=/models,readonly",
            '--mount', f"type=bind,source={config['workspace']},target=/workspace",
            '-e', 'HOME=/workspace/.home', '-e', 'XDG_CACHE_HOME=/workspace/.cache',
            '-e', 'PIPELINE_WORKSPACE=/workspace/' + WORKSPACE_NAME,
            '-e', 'PIPELINE_PYTHON=/opt/inferboost-ui/bin/python']
    if config.get('drafts'):
        args += ['--mount', f"type=bind,source={config['drafts']},target=/drafts,readonly"]
    if config.get('port') and not source:
        args += ['--publish', f"127.0.0.1:{config['port']}:9001"]
    if source:
        args += ['--mount', f'type=bind,source={source},target=/src', '--workdir', '/src']
    args += ['--entrypoint', '/usr/local/bin/inferboost-container', config['image_id'], 'idle']
    return name, args


def inspect_container(config, name):
    result = subprocess.run(command(config, 'container', 'inspect', name), capture_output=True, text=True)
    if result.returncode:
        # A daemon/auth failure must not be interpreted as an absent container.
        if 'No such' in result.stderr:
            return None
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)[0]


def ensure_container(config, source=None):
    name, options = container_options(config, source)
    existing = inspect_container(config, name)
    if existing:
        labels = existing['Config'].get('Labels', {})
        if labels.get('io.inferboost.configuration') != name or existing.get('Image') != config['image_id']:
            raise ValueError('同名容器不属于当前安装配置；保留原容器，请检查Docker上下文。')
    else:
        capture(config, *options)
    if not existing or not existing['State']['Running']:
        capture(config, 'start', name)
    return name


def install(arguments):
    parser = argparse.ArgumentParser(description='InferBoost Docker一键安装；不会安装或重配主机驱动。')
    parser.add_argument('--runtime', choices=['docker'], default='docker')
    parser.add_argument('--image', required=True, help='实际可访问的镜像引用，不假设公共Docker Hub地址')
    identity = parser.add_mutually_exclusive_group()
    identity.add_argument('--image-id', help='已导入本地镜像的完整sha256 ID')
    identity.add_argument('--digest', help='目标仓库完整sha256 manifest digest')
    parser.add_argument('--context')
    parser.add_argument('--models')
    parser.add_argument('--drafts', help='MTP不在模型根目录时的独立只读目录')
    parser.add_argument('--workspace')
    parser.add_argument('--gpus', default='all')
    parser.add_argument('--shm-size', default='16g')
    parser.add_argument('--port', type=int, help='可选：将容器9001映射到宿主127.0.0.1的这个端口')
    parser.add_argument('--bin-dir', type=Path, default=Path.home() / '.local/bin')
    parser.add_argument('--pull', choices=['missing', 'always', 'never'], default='missing')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(arguments)
    validate_host()
    if not shutil.which('docker'):
        raise ValueError('未安装Docker CLI；请先安装Docker与NVIDIA Container Toolkit。')
    config = {'schema_version': 1, 'version': VERSION, 'docker': shutil.which('docker'), 'context': args.context}
    if not (args.image_id or args.digest or '@sha256:' in args.image):
        raise ValueError('请同时提供发布记录中的 --digest 或本地 --image-id，以免使用被覆盖的标签。')
    digest = args.digest or (args.image.rsplit('@', 1)[1] if '@sha256:' in args.image else None)
    for value in [args.image_id, digest]:
        if value and not re.fullmatch(r'sha256:[0-9a-f]{64}', value):
            raise ValueError('镜像身份必须是完整sha256:加64位十六进制字符')
    if not re.fullmatch(r'all|[0-9]+(?:,[0-9]+)*', args.gpus):
        raise ValueError('GPU使用all或逗号分隔编号，例如0,1')
    if not re.fullmatch(r'[1-9][0-9]*[mMgG]', args.shm_size):
        raise ValueError('共享内存填写如16g或512m')
    if args.port and not 1 <= args.port <= 65535:
        raise ValueError('端口应为1–65535')
    print('[1/4] 检查Docker权限、架构与本地引擎', flush=True)
    engine = json.loads(capture(config, 'info', '--format', '{{json .}}'))
    if engine.get('OSType') != 'linux' or engine.get('Architecture') not in ('x86_64', 'amd64'):
        raise ValueError('Docker服务端必须为Linux amd64')
    context = json.loads(capture(config, 'context', 'inspect', *( [args.context] if args.context else [] )))[0]
    if not context['Endpoints']['docker']['Host'].startswith('unix://'):
        raise ValueError('首版安装器仅支持本机Docker引擎；远端Docker挂载路径需在远端机器安装。')
    print('[2/4] 校验并固定镜像身份', flush=True)
    pull_reference = args.image
    if digest and '@' not in pull_reference:
        # A tag before @ is accepted by Docker and the digest is authoritative.
        pull_reference += '@' + digest
    exists = subprocess.run(command(config, 'image', 'inspect', pull_reference), stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0
    if args.pull == 'always' or (not exists and args.pull == 'missing'):
        subprocess.run(command(config, 'pull', pull_reference), check=True)
    config['image_id'] = validate_image(config, pull_reference, args.image_id, digest)
    config['image'] = args.image
    config['digest'] = digest
    print('[3/4] 容器真实CUDA检查（不加载模型）', flush=True)
    gpu_arg = 'all' if args.gpus == 'all' else '"device=' + args.gpus + '"'
    subprocess.run(command(config, 'run', '--rm', '--gpus', gpu_arg, '--entrypoint', '/opt/venv/bin/python',
        config['image_id'], '/opt/pipeline/image/gpu_check.py'), check=True)
    if args.check_only:
        print('环境及CUDA检查通过；未安装启动器，未完成模型推理验收。')
        return
    def obtain(value, prompt):
        if value:
            return value
        if not sys.stdin.isatty():
            raise ValueError('非交互安装需提供' + prompt)
        return input(prompt + '：').strip()
    models = directory(obtain(args.models, '--models 主机模型根目录'))
    workspace = directory(obtain(args.workspace, '--workspace 主机持久工作目录'), create=True)
    free = shutil.disk_usage(workspace).free / 1024**3
    if free < 8:
        raise ValueError(f'工作卷可用{free:.1f}GiB；至少预留8GiB，具体模型通常需要更多。')
    config.update(models=models, workspace=workspace, drafts=directory(args.drafts) if args.drafts else None,
                  gpus=args.gpus, shm_size=args.shm_size, port=args.port)
    verify_workspace_access(config)
    print(f'[4/4] 保存启动配置；工作卷可用{free:.1f}GiB', flush=True)
    home = config_home()
    with config_lock(home):
        destination = args.bin_dir.expanduser() / 'inferboost'
        if destination.exists() and 'InferBoost host launcher' not in destination.read_text():
            raise ValueError('启动命令位置已有其他文件，未覆盖：' + str(destination))
        old = home / 'config.json'
        if old.is_file() and json.loads(old.read_text()) != config:
            backup_id = hashlib.sha256(old.read_bytes()).hexdigest()[:12]
            shutil.copy2(old, home / ('config.' + backup_id + '.json'))
        launcher_source = Path(__file__).read_text()
        launcher = home / ('runtime-' + VERSION + '.py')
        launcher.write_text(launcher_source)
        args.bin_dir.expanduser().mkdir(parents=True, exist_ok=True)
        destination.write_text('#!/usr/bin/env bash\n# InferBoost host launcher\nset -euo pipefail\nexec python3 '
                               + shlex.quote(str(launcher)) + ' "$@"\n')
        destination.chmod(0o755)
        save_json(old, config)
    print('安装完成。运行：' + str(destination))
    print('模型映射：' + models + ' → /models（只读）')
    if config['drafts']:
        print('MTP映射：' + config['drafts'] + ' → /drafts（只读）')
    print('工作目录：' + workspace + '/' + WORKSPACE_NAME + '（持久化，具体保留策略以平台为准）')
    print('PATH设置：export PATH=' + shlex.quote(str(destination.parent)) + ':"$PATH"')


def launch(arguments):
    if arguments == ['--version']:
        print('InferBoost host launcher ' + VERSION)
        return
    if arguments == ['--help']:
        print('InferBoost\n  inferboost                  打开向导；退出后容器与后台任务保留\n'
              '  inferboost status / stop    查看或停止本安装容器\n'
              '  inferboost shell            进入运行时Bash\n'
              '  inferboost dev /源码 [参数]   挂载开发源码运行quickstart\n'
              '  inferboost status/stop --dev /源码  管理对应开发容器\n'
              '  inferboost jobs watch --run-id ID  重新查看任务\n'
              '  inferboost workflow ...     转交完整流程CLI（前台运行）')
        return
    home = config_home()
    path = home / 'config.json'
    if not path.is_file():
        raise ValueError('尚未安装；请从源码目录执行 bash install.sh --help')
    config = json.loads(path.read_text())
    if config.get('schema_version') != 1 or config.get('version') != VERSION:
        raise ValueError('启动器和配置版本不匹配，请按原发布版本重新安装。')
    action = arguments[0] if arguments else 'menu'
    source = None
    if action in ('status', 'stop') and len(arguments) > 1:
        if len(arguments) != 3 or arguments[1] != '--dev':
            raise ValueError('用法：inferboost status/stop [--dev /源码路径]')
        source = directory(arguments[2])
    if action == 'dev':
        if len(arguments) < 2:
            raise ValueError('用法：inferboost dev /absolute/source [quickstart参数]')
        source = directory(arguments[1])
        if not (Path(source) / 'quickstart').is_file():
            raise ValueError('源码目录缺少quickstart')
        verify_workspace_access(dict(config, workspace=source))
        arguments = arguments[2:]
    with config_lock(home):
        name = container_name(config, source)
        if action in ('status', 'stop'):
            current = inspect_container(config, name)
            if action == 'status':
                print(json.dumps({'container': name, 'running': bool(current and current['State']['Running']),
                                  'workspace': config['workspace'], 'image_id': config['image_id']}, ensure_ascii=False, indent=2))
            elif current:
                if current['Config'].get('Labels', {}).get('io.inferboost.configuration') != name:
                    raise ValueError('容器身份不一致，未停止')
                capture(config, 'stop', '--time', '30', name, timeout=45)
                print('容器已停止；持久目录保留。下一次inferboost可重新启动并按阶段续跑。')
            return
        name = ensure_container(config, source)
    tty = sys.stdin.isatty() and sys.stdout.isatty()
    invocation = command(config, 'exec', '-i', *(['-t'] if tty else []),
                         '-e', 'TERM=' + os.environ.get('TERM', 'xterm-256color'), name)
    if action == 'shell':
        invocation += ['/bin/bash']
    elif source:
        invocation += ['/src/quickstart', *arguments]
    else:
        invocation += ['quickstart', *arguments]
    # exec client exit does not remove the detached container or its job processes.
    os.execv(config['docker'], invocation)


def main(arguments=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    try:
        if arguments and arguments[0] == 'install':
            install(arguments[1:])
        else:
            launch(arguments)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print('INFERBOOST_ERROR: ' + str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('已取消；已有配置和模型保留。', file=sys.stderr)
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
