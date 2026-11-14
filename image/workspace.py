"""Initialize a writable workspace without mixing releases or moving results."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


MANIFEST = 'image-source.json'
PREFERENCE = Path('.config/inferboost/workspace.json')
DEFAULT_WORKSPACE = '/workspace/inferboost-v050'


def fingerprints(source):
    return {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob('*'))
            if path.is_file() and path.name != MANIFEST and '__pycache__' not in path.parts}


def read_preference(preference_file):
    try:
        value = json.loads(Path(preference_file).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def resolve_destination(source, preference_file, environment):
    default = environment.get('INFERBOOST_DEFAULT_WORKSPACE', DEFAULT_WORKSPACE)
    configured = environment.get('PIPELINE_WORKSPACE') or default
    destination = Path(configured).expanduser().resolve()
    stored = read_preference(preference_file)
    if destination == Path(default).expanduser().resolve():
        expected = json.loads((Path(source) / MANIFEST).read_text())
        if stored.get('source') == expected and isinstance(stored.get('path'), str) and stored['path']:
            destination = Path(stored['path']).expanduser().resolve()
    return destination


def recent_workspaces(source, preference_file):
    stored = read_preference(preference_file)
    if stored.get('source') != json.loads((Path(source) / MANIFEST).read_text()):
        return []
    values = [stored.get('path'), *(stored.get('recent', []) if isinstance(stored.get('recent'), list) else [])]
    return list(dict.fromkeys(value for value in values if isinstance(value, str) and value))[:8]


def remember(source, destination, preference_file):
    """Update preferences atomically; independent terminals cannot share a temp file."""
    preference_file = Path(preference_file)
    value = {'path': str(destination), 'source': json.loads((Path(source) / MANIFEST).read_text()),
             'recent': list(dict.fromkeys([str(destination), *recent_workspaces(source, preference_file)]))[:8]}
    preference_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=preference_file.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        temporary.replace(preference_file)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_destination(source, destination, current=None):
    source, destination = Path(source).resolve(), Path(destination).expanduser().resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError('工作目录不能与镜像源码目录重叠，请选择独立的新目录。')
    if current is not None:
        current = Path(current).resolve()
        if destination != current and (current in destination.parents or destination in current.parents):
            raise ValueError('新旧工作目录不能互相嵌套，请在同一挂载盘中使用并列目录。')
    if any((parent / MANIFEST).is_file() for parent in destination.parents):
        raise ValueError('不能在已有工作目录内部创建另一个工作目录，请选择并列目录。')
    if destination.exists() and not destination.is_dir():
        raise ValueError('工作目录路径是文件，请选择目录。')
    if not destination.exists() or not any(destination.iterdir()):
        return destination
    marker = destination / MANIFEST
    try:
        actual = json.loads(marker.read_text()) if marker.is_file() else None
    except ValueError:
        actual = None
    expected = json.loads((source / MANIFEST).read_text())
    if actual != expected:
        raise ValueError('工作目录属于其他代码版本或不是镜像工作目录。原结果不会被覆盖；'
                         '请选新空目录，旧量化结果可在主菜单中复用。')
    for name, checksum in expected.items():
        path = destination / name
        if not path.resolve().is_relative_to(destination) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError('镜像代码或公共数据已修改: ' + name + '。'
                             '开发者可直接运行该目录的 ./quickstart；标准镜像入口请使用新的工作目录。')
    return destination


def initialize(source, destination, current=None):
    source = Path(source).resolve()
    destination = validate_destination(source, destination, current)
    if destination.exists() and any(destination.iterdir()):
        return destination
    expected = json.loads((source / MANIFEST).read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stage application files only. A failed copy never leaves a half-initialized
    # workspace; POSIX rename refuses to replace a nonempty destination.
    with tempfile.TemporaryDirectory(prefix='.inferboost-init-', dir=destination.parent) as staging:
        prepared = Path(staging) / 'workspace'
        prepared.mkdir()
        for name, checksum in expected.items():
            original = source / name
            target = prepared / name
            if not original.resolve().is_relative_to(source) or not target.resolve().is_relative_to(prepared):
                raise ValueError('镜像清单包含越界路径: ' + name)
            if hashlib.sha256(original.read_bytes()).hexdigest() != checksum:
                raise ValueError('镜像源文件校验失败: ' + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
        shutil.copy2(source / MANIFEST, prepared / MANIFEST)
        # Respect normal directory permissions instead of the staging dir's 0700.
        prepared.chmod(0o755)
        try:
            os.rename(prepared, destination)
        except OSError:
            # Another terminal may have completed initialization while we copied.
            validate_destination(source, destination, current)
            if not (destination / MANIFEST).is_file():
                raise
    return destination


if __name__ == '__main__':
    if sys.argv[1] == 'manifest':
        source = Path(sys.argv[2])
        (source / MANIFEST).write_text(json.dumps(fingerprints(source), sort_keys=True, indent=2) + '\n')
    else:
        try:
            initialize(sys.argv[1], sys.argv[2])
        except (ValueError, OSError) as error:
            raise SystemExit('WORKSPACE_ERROR: ' + str(error))
