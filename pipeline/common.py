import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}', value):
        raise ValueError('标识只能使用字母、数字、下划线、点和短横线，长度1–96')
    return value


def inside(root, path):
    path = Path(path).resolve()
    if Path(root).resolve() not in path.parents:
        raise ValueError(f'输出必须在工作目录内部: {path}')
    return path


def capture(command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs).strip()


def runtime_environment(root, config):
    cache = inside(root, Path(root) / 'cache')
    temporary = inside(root, Path(root) / 'tmp')
    temporary.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(config['runtime'].get('environment', {}))
    if not environment.get('VLLM_RPC_BASE_PATH'):
        from .ipc import socket_directory
        environment['VLLM_RPC_BASE_PATH'] = socket_directory(root)
    environment.update(
        CUDA_VISIBLE_DEVICES=str(config['runtime'].get('gpu', 0)),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        TMPDIR=str(temporary), XDG_CACHE_HOME=str(cache / 'xdg'),
        HF_HOME=str(cache / 'huggingface'), CUDA_CACHE_PATH=str(cache / 'cuda'),
        VLLM_CACHE_ROOT=str(cache / 'vllm'), TRITON_CACHE_DIR=str(cache / 'triton'),
        TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TORCH_EXTENSIONS_DIR=str(cache / 'torch_extensions'),
        FLASHINFER_WORKSPACE_BASE=str(cache / 'flashinfer'),
        MAX_JOBS=str(config['runtime'].get('max_jobs', 2)), FLASHINFER_NVCC_THREADS='1',
        TORCHINDUCTOR_COMPILE_THREADS=str(config['runtime'].get('max_jobs', 2)),
        PYTHONPATH=str(root), VLLM_SERVER_DEV_MODE='1')
    return environment


def source_digest(root):
    files = sorted(Path(root).glob('pipeline/**/*.py')) + sorted(Path(root).glob('scripts/*.py'))
    return {str(path.relative_to(root)): digest(path) for path in files}


def load_config(root, path):
    config = read_json(path)
    if config.get('schema_version') != 1:
        raise ValueError('需要 schema_version=1')
    quant = config['quantization']
    if quant.get('mode') not in ('reuse', 'fresh') or quant.get('method') != 'awq4':
        raise ValueError('本版支持 AWQ4 的 fresh/reuse；其他算法尚未验收')
    if not Path(config['runtime']['python']).is_absolute():
        raise ValueError('runtime.python 必须是隔离环境Python的绝对路径')
    names = [identifier(profile['id']) for profile in config['profiles']]
    if not names or len(names) != len(set(names)):
        raise ValueError('profiles 必须非空且 id 唯一')
    for profile in config['profiles']:
        if profile['attention'] not in ('fa2', 'fa3', 'flashinfer'):
            raise ValueError('未知 attention 后端')
        if profile.get('mtp', 0) not in (0, 1, 2):
            raise ValueError('本版只验证 MTP0/1/2')
        if profile.get('mtp') and not config.get('mtp_model'):
            raise ValueError('MTP方案需要 mtp_model')
    benchmark = config['benchmark']
    if not benchmark['concurrencies'] or any(type(n) is not int or n < 1 for n in benchmark['concurrencies']):
        raise ValueError('concurrencies 必须为正整数列表')
    if max(benchmark['concurrencies']) > config['server']['max_num_seqs']:
        raise ValueError('本版闭环测试并发不能超过 max_num_seqs')
    if benchmark.get('output_mode', 'natural') not in ('natural', 'fixed'):
        raise ValueError('output_mode 仅允许 natural/fixed')
    if benchmark.get('cache_modes', ['cold']) not in (['cold'], ['cold', 'warm']):
        raise ValueError('cache_modes 只支持 cold 或 cold,warm 成对测试')
    return config
