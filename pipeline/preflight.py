"""Cheap read-only capability discovery; estimates are never GPU acceptance."""
import csv
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .common import read_json

GIB = 1024 ** 3


def memory_snapshot():
    values = {}
    try:
        values = {line.split(':')[0]: int(line.split()[1]) * 1024
                  for line in Path('/proc/meminfo').read_text().splitlines()}
    except (OSError, ValueError):
        pass
    limit, used = values.get('MemTotal'), None
    for maximum, current in [('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                             ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        try:
            cap = int(Path(maximum).read_text())
            if cap < 2 ** 60 and (limit is None or cap <= limit):
                limit, used = cap, int(Path(current).read_text())
        except (OSError, ValueError):
            continue
    return {'limit_gib': limit / GIB if limit else None,
            'used_gib': used / GIB if used is not None else None,
            'available_gib': (min(limit - used, values.get('MemAvailable', limit)) / GIB
                              if used is not None else values.get('MemAvailable', 0) / GIB or None)}


def gpu_snapshot():
    command = shutil.which('nvidia-smi')
    if not command:
        return []
    fields = 'index,name,memory.total,memory.used,utilization.gpu,driver_version,compute_cap'
    try:
        result = subprocess.run([command, '--query-gpu=' + fields, '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode:
            return []
        rows = []
        for row in csv.reader(result.stdout.splitlines()):
            rows.append({'index': int(row[0]), 'name': row[1].strip(), 'total_gib': float(row[2]) / 1024,
                         'used_gib': float(row[3]) / 1024, 'utilization': row[4].strip(),
                         'driver': row[5].strip(), 'capability': [int(n) for n in row[6].strip().split('.')]})
        return rows
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return []


def mount_info(path, mountinfo='/proc/self/mountinfo'):
    path = Path(path).resolve()
    matches = []
    try:
        for line in Path(mountinfo).read_text().splitlines():
            left, right = line.split(' - ', 1)
            fields = left.split()
            mount = Path(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), fields[4]))
            if path == mount or mount in path.parents:
                matches.append((len(str(mount)), str(mount), right.split()[0]))
    except (OSError, ValueError, IndexError):
        pass
    if not matches:
        return {'mount': None, 'filesystem': None, 'persistence': '未知，请核对平台持久卷配置'}
    _, mount, filesystem = max(matches)
    hint = ('容器临时文件系统，重建可能丢失' if filesystem in ('overlay', 'tmpfs', 'ramfs')
            else '已发现挂载；跨实例保留仍需平台确认')
    return {'mount': mount, 'filesystem': filesystem, 'persistence': hint}


def storage_snapshot(path, check_write=False):
    path = Path(path).expanduser().resolve()
    if check_write:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path) as stream:
            stream.write(b'workspace-check')
    existing = path
    while not existing.exists():
        existing = existing.parent
    return {'path': str(path), 'free_gib': shutil.disk_usage(existing).free / GIB, **mount_info(path)}


def workspace_choices():
    return [path for path in (Path('/vepfs'), Path('/mnt'), Path('/workspace'))
            if path.is_dir() and os.access(path, os.W_OK)]


def model_summary(path):
    from .models import inspect_model
    from .quantization_catalog import ADAPTERS
    path = Path(path)
    if not any((path / name).is_file() for name in ['tokenizer.json', 'tokenizer.model']):
        raise ValueError('模型目录缺少tokenizer.json或tokenizer.model，请挂载完整模型。')
    identity, tensors = inspect_model(path)
    config = identity['config']
    text = config.get('text_config', config)
    model_type = text.get('model_type', config.get('model_type'))
    adapter = ADAPTERS.get(config.get('model_type')) or ADAPTERS.get(model_type)
    mtp = [name for name in tensors if re.search(r'(^|\.)mtp[._]', name)]
    return {'path': str(Path(path).resolve()), 'model_type': model_type,
            'precision': text.get('dtype', text.get('torch_dtype', config.get('torch_dtype', '以权重为准'))),
            'quantized': bool(config.get('quantization_config')), 'mtp_tensors': len(mtp),
            'max_model_len': text.get('max_position_embeddings'),
            'weight_gib': sum(part['bytes'] for part in identity['shards'].values()) / GIB,
            'adapter': adapter, 'config': config, 'shards': len(identity['shards'])}


def recipe_choices(model, gpus):
    from .quantization_catalog import FRESH_RECIPES, RECIPES, eligibility
    from .quantization_labels import describe_quantization
    capability = min((gpu['capability'][0] * 10 + gpu['capability'][1] for gpu in gpus), default=None)
    rows = []
    for name in FRESH_RECIPES:
        recipe = RECIPES[name]
        reason = None
        if not model['adapter']:
            reason = '当前没有已接入的模型适配器'
        elif capability is None:
            reason = '未检测到GPU能力，请检查驱动和设备挂载'
        elif capability < recipe.minimum_sm:
            reason = f'此配方要求SM{recipe.minimum_sm}及以上，当前SM{capability}'
        else:
            reason = eligibility(name, {'model_type': model['config'].get('model_type'),
                                       'adapter': model['adapter']}, [capability // 10, capability % 10])['reason']
        rows.append({'id': name.removeprefix('compressor_'),
                     'label': describe_quantization(name)['quantization_label'], 'reason': reason,
                     'enabled': reason is None, 'validation': '通过静态准入；本模型仍需实测'})
    return rows


def inspect_resources(root, check_write=False):
    return {'storage': storage_snapshot(root, check_write=check_write),
            'memory': memory_snapshot(), 'gpus': gpu_snapshot()}


def validate_resources(resources, devices, model, recipes, skip_quantization=False):
    errors, warnings = [], []
    available = {gpu['index']: gpu for gpu in resources['gpus']}
    if any(device not in available for device in devices):
        errors.append('所选GPU不存在或nvidia-smi不可用，请检查设备挂载与驱动。')
    if not skip_quantization and not model['adapter']:
        errors.append('模型架构尚未接入量化适配器：' + model['model_type'])
    memory = resources['memory']
    if not skip_quantization and memory['limit_gib'] is not None and memory['limit_gib'] < 32:
        errors.append('当前量化配置至少需要32 GiB容器内存。')
    if any(available[device]['used_gib'] > 1 for device in devices if device in available):
        warnings.append('所选GPU已有显存占用；正式运行还会执行空闲检查。')
    # File lower bound excludes caches and protected full-precision layers.
    ratio = sum(.5 if name in ('gptq8', 'w8a8', 'fp8') else .25 for name in recipes)
    weight_floor = model['weight_gib'] * ratio if not skip_quantization else 0
    estimate = weight_floor * 1.5 + 8
    free = resources['storage']['free_gib']
    if free < weight_floor:
        errors.append(f'剩余磁盘{free:.1f} GiB，小于候选权重粗略下限{weight_floor:.1f} GiB。')
    elif free < estimate:
        warnings.append(f'磁盘建议至少约{estimate:.1f} GiB（粗估，另需编译与校准缓存），当前{free:.1f} GiB。')
    if resources['storage']['filesystem'] in ('overlay', 'tmpfs', 'ramfs'):
        warnings.append('当前工作目录位于临时文件系统，请优先选择持久卷。')
    return {'errors': errors, 'warnings': warnings, 'estimated_disk_gib': estimate,
            'estimate_is_guarantee': False}
