"""Small, versioned configuration for the vLLM 0.33 inference adapter."""
import copy
import json
from pathlib import Path

from pipeline.common import runtime_environment


ADAPTER_VERSION = 'vllm-0.33-stage2-v1'


def load(root, path):
    supplied = json.loads(Path(path).read_text())
    config = {
        'schema_version': 1,
        'runtime': copy.deepcopy(supplied['runtime']),
        'server': {'port': 18550, 'max_model_len': 32768, 'max_num_seqs': 4,
                   'gpu_memory_utilization': .9, 'startup_timeout_seconds': 2400},
        'benchmark': {'dataset': str(root / 'data/public100.json'), 'max_tokens': 256,
                      'timeout_seconds': 300, 'deadline_seconds': 600},
        'quality': {'dataset': str(root / 'data/quality100.json') if (root / 'data/quality100.json').is_file() else None},
        'search': {'strategy': 'staged', 'attention': ['fa2', 'fa3', 'flashinfer'],
                   'kernels': ['marlin', 'auto', 'humming'],
                   'graphs': ['none', 'piecewise', 'full'], 'budgets': [2048, 4096, 8192],
                   'mtp_depths': [0, 1, 2], 'screening_samples': 20, 'finalists': 2,
                   'repeats': 3, 'max_trials': 24, 'objective_cache': 'warm'},
    }
    for key in ['server', 'benchmark', 'search', 'quality']:
        config[key].update(supplied.get(key, {}))
    if 'experience' in supplied:
        config['experience'] = copy.deepcopy(supplied['experience'])
    config['benchmark']['concurrencies'] = [1]
    config['benchmark']['cache_modes'] = ['cold', 'warm']
    config['benchmark']['dataset'] = str((root / config['benchmark']['dataset']).resolve())
    if config['quality']['dataset']:
        config['quality']['dataset'] = str((root / config['quality']['dataset']).resolve())
    validate(config)
    return config


def validate(config):
    if not Path(config['runtime']['python']).is_absolute():
        raise ValueError('推理 Python 必须是绝对路径')
    search = config['search']
    allowed = {'attention': {'fa2', 'fa3', 'flashinfer'},
               'kernels': {'auto', 'marlin', 'humming'},
               'graphs': {'none', 'piecewise', 'full'}, 'mtp_depths': {0, 1, 2}}
    for name, values in allowed.items():
        if not search[name] or len(search[name]) != len(set(search[name])) or not set(search[name]) <= values:
            raise ValueError('无效或重复的搜索选项: ' + name)
    if search['strategy'] not in ('staged', 'full') or search['objective_cache'] not in ('cold', 'warm'):
        raise ValueError('搜索模式或目标缓存状态无效')
    for name in ['screening_samples', 'finalists', 'repeats', 'max_trials']:
        if type(search[name]) is not int or search[name] < 1:
            raise ValueError(name + ' 必须是正整数')
    for budget in search['budgets']:
        if type(budget) is not int or budget < config['server']['max_num_seqs']:
            raise ValueError('调度预算必须为整数，且不小于 max_num_seqs')
    if not search['budgets'] or len(search['budgets']) != len(set(search['budgets'])):
        raise ValueError('调度预算不能为空或重复')
    if not 0 < config['server']['gpu_memory_utilization'] < 1:
        raise ValueError('gpu_memory_utilization 必须介于0和1之间')


def environment(root, config, source=None):
    env = runtime_environment(root, config)
    paths = [str(source or root)]
    if config['runtime'].get('benchmark_site'):
        paths.append(config['runtime']['benchmark_site'])
    env['PYTHONPATH'] = ':'.join(paths)
    env['NO_PROXY'] = env['no_proxy'] = '127.0.0.1,localhost'
    # Stage two preserves checkpoint precision rather than introducing a new quantization.
    for name in ['VLLM_HUMMING_INPUT_QUANT_CONFIG', 'VLLM_MARLIN_INPUT_DTYPE']:
        env.pop(name, None)
    return env
