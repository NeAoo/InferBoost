"""Generate bounded, compatible configurations without altering checkpoint precision."""
import hashlib
import itertools
import json


ATTENTION_LABELS = {'fa2': 'FA2', 'fa3': 'FA3', 'flashinfer': 'FlashInfer'}
GRAPH_LABELS = {'none': '无Graph', 'piecewise': '分段Graph', 'full': '完整+分段Graph'}


def make_trial(**values):
    trial = {'attention': 'fa2', 'kernel': 'marlin', 'compile': True,
             'graph': 'full', 'budget': 8192, 'mtp': 0}
    trial.update(values)
    if trial['graph'] != 'none' and not trial['compile']:
        raise ValueError('本适配器只允许无编译时关闭Graph；编译和Graph是独立选项')
    fingerprint = hashlib.sha256(json.dumps(trial, sort_keys=True).encode()).hexdigest()[:12]
    trial['id'] = 't-' + fingerprint
    graph = GRAPH_LABELS[trial['graph']] if trial['compile'] else 'Eager'
    trial['label'] = ' · '.join([ATTENTION_LABELS[trial['attention']], trial['kernel'], graph,
                                'MTP' + str(trial['mtp']), str(trial['budget'])])
    return trial


def replace(trial, **changes):
    values = {key: trial[key] for key in ['attention', 'kernel', 'compile', 'graph', 'budget', 'mtp']}
    values.update(changes)
    return make_trial(**values)


def capabilities(model, hardware, config):
    search = config['search']
    sm = hardware['capability'][0] * 10 + hardware['capability'][1]
    allowed = {'attention': [], 'kernels': [], 'mtp_depths': [0]}
    skipped = []
    for attention in search['attention']:
        reason = None
        if attention == 'fa3' and hardware['capability'][0] != 9:
            reason = 'FA3要求Hopper，当前不是SM90'
        if attention == 'flashinfer' and not hardware.get('flashinfer'):
            reason = '当前推理环境未安装FlashInfer'
        if reason:
            skipped.append({'option': attention, 'reason': reason})
        else:
            allowed['attention'].append(attention)
    for kernel in search['kernels']:
        reason = None
        if model['method'] not in ('awq', 'awq-marlin', 'gptq', 'gptq-marlin', 'compressed-tensors') and kernel != 'auto':
            reason = '该权重格式目前仅开放原生auto，其他矩阵后端需单独适配'
        if kernel == 'marlin' and (model['activation_bits'] != 16 or model['bits'] not in (4, 8)):
            reason = '本适配器的Marlin候选仅用于W4/W8 A16，不能改变已选模型的激活精度'
        if kernel == 'humming' and (not hardware.get('humming') or sm < 75):
            reason = 'Humming未安装或GPU不满足SM75要求'
        if model['method'] == 'none' and kernel != 'auto':
            reason = '未量化模型不使用本轮的量化矩阵内核候选'
        if reason:
            skipped.append({'option': kernel, 'reason': reason})
        else:
            allowed['kernels'].append(kernel)
    if model.get('mtp_supported'):
        allowed['mtp_depths'] = list(dict.fromkeys([0, *search['mtp_depths']]))
    elif any(search['mtp_depths']):
        skipped.append({'option': 'MTP', 'reason': '没有经过适配的匹配BF16草稿权重'})
    if model['activation_bits'] == 8 and model['activation_type'] == 'float':
        minimum_sm = 90 if model['bits'] == 4 else 89
        if sm < minimum_sm:
            raise ValueError(f'当前W{model["bits"]} AFP8适配要求至少SM{minimum_sm}；不退回A16冒充原量化方案')
    if not allowed['attention'] or not allowed['kernels']:
        raise ValueError('没有与当前模型/GPU兼容的基础候选')
    return {'allowed': allowed, 'skipped': skipped}


def baseline(capability, config):
    allowed = capability['allowed']
    kernel = 'marlin' if 'marlin' in allowed['kernels'] else 'auto'
    attention = 'fa2' if 'fa2' in allowed['attention'] else allowed['attention'][0]
    return make_trial(attention=attention, kernel=kernel, budget=config['search']['budgets'][-1])


def unique(trials):
    return list({trial['id']: trial for trial in trials}.values())


def full_grid(capability, config):
    allowed = capability['allowed']
    graph_options = [(True, graph) for graph in config['search']['graphs']] + [(False, 'none')]
    trials = [make_trial(attention=attention, kernel=kernel, compile=compile_, graph=graph,
                         budget=budget, mtp=mtp)
              for attention, kernel, (compile_, graph), budget, mtp in itertools.product(
                  allowed['attention'], allowed['kernels'], graph_options,
                  config['search']['budgets'], allowed['mtp_depths'])]
    if len(trials) > config['search']['max_trials']:
        raise ValueError(f'完整搜索有{len(trials)}组，超过max_trials；请缩小范围或显式提高上限')
    return unique(trials)


def screen_phase(name, parents, capability, config):
    allowed = capability['allowed']
    trials = []
    if name == 'attention':
        trials = [replace(parents[0], attention=value) for value in allowed['attention']]
    elif name == 'kernels':
        trials = [replace(parent, kernel=value) for parent in parents for value in allowed['kernels']]
    elif name == 'graphs':
        trials = [replace(parent, graph=value, compile=True) for parent in parents for value in config['search']['graphs']]
        trials += [replace(parent, graph='none', compile=False) for parent in parents]
    elif name == 'budgets':
        trials = [replace(parent, budget=value) for parent in parents for value in config['search']['budgets']]
    elif name == 'mtp':
        trials = [replace(parent, mtp=value) for parent in parents for value in allowed['mtp_depths']]
        # Keep a full-graph FlashInfer MTP seed even if short screening favoured another graph.
        if 'flashinfer' in allowed['attention'] and 2 in allowed['mtp_depths']:
            trials.append(replace(parents[0], attention='flashinfer', graph='full', compile=True, mtp=2))
    else:
        raise ValueError('未知筛选阶段: ' + name)
    return unique(trials)


def server_command(config, model, trial, port=None, host='127.0.0.1'):
    server = config['server']
    attention = {'backend': 'FLASHINFER'} if trial['attention'] == 'flashinfer' else {
        'backend': 'FLASH_ATTN', 'flash_attn_version': 3 if trial['attention'] == 'fa3' else 2}
    command = [config['runtime']['python'], '-m', 'vllm.entrypoints.openai.api_server',
               '--model', model['load_view'], '--host', host, '--port', str(port or server['port']),
               '--served-model-name', 'pipeline-model', '--dtype', 'auto', '--tensor-parallel-size', '1',
               '--max-model-len', str(server['max_model_len']), '--max-num-seqs', str(server['max_num_seqs']),
               '--max-num-batched-tokens', str(trial['budget']),
               '--gpu-memory-utilization', str(server['gpu_memory_utilization']),
               '--enable-prefix-caching', '--enable-chunked-prefill', '--language-model-only',
               '--kv-cache-dtype', 'auto', '--attention-config', json.dumps(attention),
               '--worker-cls', 'pipeline.inference.audit.Worker']
    if 'qwen3' in str(model['model_type']):
        command += ['--reasoning-parser', 'qwen3']
    if model['method'] == 'awq':
        command += ['--quantization', 'awq_marlin']
    elif model['method'] in ('gptq', 'gptq-marlin'):
        command += ['--quantization', 'gptq_marlin']
    command += ['--kernel-config', json.dumps({'linear_backend': trial['kernel'], 'moe_backend': trial['kernel']})]
    if trial['compile']:
        graph = {'none': 'NONE', 'piecewise': 'PIECEWISE', 'full': 'FULL_AND_PIECEWISE'}[trial['graph']]
        command += ['--compilation-config', json.dumps({'mode': 3, 'cudagraph_mode': graph})]
    else:
        command += ['--enforce-eager']
    if trial['mtp']:
        if not model.get('mtp_supported'):
            raise ValueError('该模型没有已适配MTP')
        command += ['--speculative-config', json.dumps({'method': 'mtp', 'model': model['references']['mtp'],
                                                       'num_speculative_tokens': trial['mtp']})]
    return command
