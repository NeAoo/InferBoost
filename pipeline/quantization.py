"""Stage one: architecture admission, independent backends, real quality comparison."""
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import threading
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import time

from . import benchmark, models, quality, runtime
from .terminal import Progress
from .common import digest, identifier, inside, read_json, runtime_environment, source_digest, write_json
from .quantization_catalog import FRESH_RECIPES, RECIPES, eligibility, inspect_source
from .calibration import assert_independent, load_calibration, official_defaults

DEFAULT_MAX_DROP_PP = 5.0
DEFAULT_MAX_REGRESSIONS = 3


def load_config(root, path):
    config = read_json(path)
    if config.get('schema_version') != 2 or config.get('stage') != 'quantization':
        raise ValueError('第一阶段配置需要 schema_version=2、stage=quantization')
    if config.get('mtp_policy', 'preserve_original_precision') != 'preserve_original_precision':
        raise ValueError('本组件只量化主模型，MTP必须保留原始精度')
    config['mtp_policy'] = 'preserve_original_precision'
    calibration = config.setdefault('calibration', official_defaults())
    if calibration.get('mode') == 'official':
        calibration = {**official_defaults(), **calibration}
        config['calibration'] = calibration
        from .calibration import OFFICIAL_DATASET
        if calibration['dataset'] != OFFICIAL_DATASET or not re.fullmatch(r'[0-9a-f]{40}', calibration['revision']):
            raise ValueError('官方校准模式需要PerfectBlend及固定revision SHA')
        if any(type(calibration[key]) is not int or calibration[key] <= 0 for key in ['count', 'max_tokens']):
            raise ValueError('官方校准数量和长度必须为正整数')
        if type(calibration['seed']) is not int or calibration['seed'] < 0:
            raise ValueError('校准seed必须是非负整数')
    elif calibration.get('mode') not in ('frozen', 'synthetic'):
        raise ValueError('未知校准模式')
    for field in ['source']:
        config[field] = str((Path(root) / config[field]).resolve())
    if not config.get('datasets'):
        raise ValueError('至少需要一份评测数据')
    names = set()
    for dataset in config['datasets']:
        identifier(dataset['id'])
        if dataset['id'] in names:
            raise ValueError('数据集id重复')
        names.add(dataset['id'])
        dataset['path'] = str((Path(root) / dataset['path']).resolve())
        quality.load_dataset(dataset['path'])
    candidates = config.get('candidates', [])
    if not any(item.get('recipe') == 'original' for item in candidates):
        candidates.insert(0, {'id': 'original', 'recipe': 'original', 'mode': 'reuse'})
    config['candidates'] = sorted(candidates, key=lambda item: item.get('recipe') != 'original')
    ids = [identifier(item['id']) for item in candidates]
    if len(ids) != len(set(ids)) or sum(item['recipe'] == 'original' for item in candidates) != 1:
        raise ValueError('候选id必须唯一，且只能有一个原模型参照')
    for item in candidates:
        if item['recipe'] not in RECIPES or item.get('mode', 'fresh') not in ('fresh', 'reuse'):
            raise ValueError('未知配方或执行模式')
        if RECIPES[item['recipe']].backend == 'gptqmodel' and item.get('mode', 'fresh') == 'fresh':
            raise ValueError('新量化已统一LLM Compressor，请使用compressor_awq4/compressor_gptq4/compressor_gptq8；GPTQModel旧产物仅支持reuse对照')
        if item.get('model'):
            item['model'] = str((Path(root) / item['model']).resolve())
    config.setdefault('benchmark', {})
    config['benchmark'].setdefault('concurrencies', [1])
    config['benchmark'].setdefault('cache_modes', ['cold', 'warm'])
    config['benchmark'].setdefault('max_tokens', 128)
    repeats = config['benchmark'].get('performance_repeats', 1)
    if type(repeats) is not int or not 1 <= repeats <= 20:
        raise ValueError('performance_repeats必须在1–20之间')
    config['benchmark']['output_mode'] = 'natural'
    if config['benchmark']['cache_modes'] not in (['cold'], ['cold', 'warm']):
        raise ValueError('缓存测试需为cold或cold,warm')
    concurrencies = config['benchmark']['concurrencies']
    if not concurrencies or any(type(n) is not int or n <= 0 for n in concurrencies) or 1 not in concurrencies:
        raise ValueError('第一阶段并发需要正整数列表，并包含1用于质量对照')
    if max(concurrencies) > config['server']['max_num_seqs']:
        raise ValueError('评测并发超过服务max_num_seqs')
    config.setdefault('quality_gate', {'max_drop_pp': DEFAULT_MAX_DROP_PP,
                                       'max_regressions': DEFAULT_MAX_REGRESSIONS})
    gate = config['quality_gate']
    if not isinstance(gate, dict):
        raise ValueError('quality_gate必须是质量门槛对象')
    drop, regressions = gate.get('max_drop_pp', 0), gate.get('max_regressions', 0)
    if (type(drop) not in (int, float) or not math.isfinite(drop) or drop < 0
            or type(regressions) is not int or regressions < 0):
        raise ValueError('总分下降阈值必须为有限非负数，新增错题阈值必须为非负整数')
    devices = config['runtime'].get('quantization_gpus', [config['runtime'].get('gpu', 0)])
    if not devices or any(type(value) is not int or value < 0 for value in devices) or len(devices) != len(set(devices)):
        raise ValueError('quantization_gpus必须是非空、不重复的GPU编号')
    if config['runtime'].get('gpu', 0) not in devices:
        raise ValueError('性能评测GPU必须包含在申请的量化GPU列表中')
    return config


def plan(root, config, capability=None):
    model = inspect_source(Path(config['source']))
    candidates = []
    for item in config['candidates']:
        admission = eligibility(item['recipe'], model, capability)
        backend = RECIPES[item['recipe']].backend
        if backend != 'none' and item.get('mode', 'fresh') == 'fresh':
            python = config.get('backends', {}).get(backend, {}).get('python')
            if not python or not Path(python).is_file():
                admission.update(eligible=False, reason='未配置独立量化环境: ' + backend)
        candidates.append({**item, **admission})
    return {'model': model, 'candidates': candidates, 'datasets': config['datasets'],
            'quality_gate': config['quality_gate'], 'mtp_policy': config['mtp_policy'],
            'fixed_inference': {'mtp': 0, 'attention': config.get('reference_profile', {}).get('attention', 'fa2'),
                                'graph': config.get('reference_profile', {}).get('graph', True)},
            'note': '可执行不等于该模型已通过精度验证；每份数据分别对照原模型'}


def environment_fingerprints(root, config):
    environments = {'inference': {'python': config['runtime']['python'], 'environment': config['runtime'].get('environment', {})},
                    **config.get('backends', {})}
    code = ('import json,sys; from importlib.metadata import version,PackageNotFoundError\n'
            'result={"python":sys.version,"executable":sys.executable}\n'
            'for package in ["torch","transformers","vllm","llmcompressor","compressed-tensors","gptqmodel","datasets","accelerate"]:\n'
            ' try: result[package]=version(package)\n'
            ' except PackageNotFoundError: result[package]=None\n'
            'print(json.dumps(result))')
    fingerprints = {}
    for name, settings in environments.items():
        if not settings.get('python') or not Path(settings['python']).is_file():
            fingerprints[name] = {'available': False}
            continue
        environment = runtime_environment(root, config)
        environment.update(settings.get('environment', {}))
        result = subprocess.run([settings['python'], '-c', code], env=environment, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError('环境版本探测失败: ' + name)
        fingerprints[name] = {'available': True, **json.loads(result.stdout.splitlines()[-1])}
    return fingerprints


def tensor_fingerprints(directory, names):
    """Hash the real payload, not just metadata, without importing a GPU framework."""
    directory = Path(directory)
    index = directory / 'model.safetensors.index.json'
    mapping = read_json(index)['weight_map'] if index.exists() else {name: 'model.safetensors' for name in names}
    groups = {}
    for name in names:
        if name not in mapping:
            raise ValueError('MTP权重缺失: ' + name)
        groups.setdefault(mapping[name], []).append(name)
    output = {}
    for filename, keys in groups.items():
        if Path(filename).name != filename:
            raise ValueError('分片路径非法')
        with (directory / filename).open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(length))
            for key in keys:
                if key not in header:
                    raise ValueError('MTP权重缺失: ' + key)
                item = header[key]
                start, end = item['data_offsets']
                stream.seek(8 + length + start)
                hasher = hashlib.sha256()
                remaining = end - start
                while remaining:
                    block = stream.read(min(remaining, 8 * 1024**2))
                    if not block:
                        raise ValueError('MTP分片残缺')
                    hasher.update(block)
                    remaining -= len(block)
                output[key] = {'dtype': item['dtype'], 'shape': item['shape'], 'sha256': hasher.hexdigest()}
    return output


def run_logged(command, environment, log, root, cancelled=None):
    """Own the complete subprocess group and stop it if the coordinator stops."""
    with Path(log).open('a') as stream:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            labels = {'quantization.log': '量化', 'calibration.log': '准备校准集',
                      'asset_preparation.log': '核验模型'}
            label = labels.get(Path(log).name, Path(log).stem)
            with Progress(Path(log).parent.name + ' · ' + label, live=False) as progress:
                while process.poll() is None:
                    detail = '日志：' + str(log)
                    backend_progress = Path(log).parent / 'backend_progress.json'
                    if Path(log).name == 'quantization.log' and backend_progress.is_file():
                        try:
                            detail = read_json(backend_progress).get('message', detail)
                        except (OSError, ValueError):
                            pass  # A writer may be between truncate and replace.
                    progress.update(detail=detail)
                    if cancelled is not None and cancelled.wait(.5):
                        raise RuntimeError('任务已取消')
                    if cancelled is None:
                        time.sleep(.5)
                if process.returncode != 0:
                    raise RuntimeError('子任务失败，查看 ' + str(log))
        finally:
            # A completed parent can still leave a compiler child behind.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


def prepare_calibration(root, config, run_dir, environment):
    settings = config.get('calibration', official_defaults())
    if settings.get('mode') == 'frozen':
        directory = (Path(root) / settings['path']).resolve()
    elif settings.get('mode') == 'official':
        settings = {**official_defaults(), **settings}
        directory = run_dir / 'calibration'
        backend = config['backends']['llmcompressor']
        preparation_environment = environment.copy()
        preparation_environment.update(backend.get('environment', {}))
        preparation_environment.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='0', HF_DATASETS_OFFLINE='0',
                                       HF_DATASETS_CACHE=str(root / 'cache/datasets'), OMP_NUM_THREADS='2')
        command = [backend['python'], str(root / 'scripts/prepare_official_calibration.py'),
                    '--model', config['source'], '--output', str(directory),
                    '--test-data', *[item['path'] for item in config['datasets']],
                    '--count', str(settings['count']), '--max-tokens', str(settings['max_tokens']),
                    '--seed', str(settings['seed']), '--revision', settings['revision']]
        if settings.get('raw_data'):
            command += ['--raw-data', str((Path(root) / settings['raw_data']).resolve())]
            preparation_environment.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        run_logged(command,
                   preparation_environment, run_dir / 'calibration.log', root)
    elif settings.get('mode') == 'synthetic':
        directory = run_dir / 'calibration'
        python = config['runtime']['python']
        run_logged([python, str(root / 'scripts/prepare_calibration.py'), '--model', config['source'],
                    '--output', str(directory), '--test-data', config['datasets'][0]['path'],
                    '--count', str(settings.get('count', 64)), '--max-tokens', str(settings.get('max_tokens', 512)),
                    '--profile', settings.get('profile', 'mixed')],
                   environment, run_dir / 'calibration.log', root)
    else:
        raise ValueError('校准模式仅支持官方official、已冻结frozen或明确选择的synthetic工程数据')
    manifest = read_json(directory / 'manifest.json')
    if manifest['tokenizer_sha256'] != digest(Path(config['source']) / 'tokenizer.json'):
        raise ValueError('校准tokenizer不匹配')
    for name, key in [('cases.jsonl', 'cases_sha256'), ('tokens.jsonl', 'tokens_sha256')]:
        if digest(directory / name) != manifest[key]:
            raise ValueError('校准文件SHA256不匹配')
    calibration_queries = {json.dumps(row['messages'], sort_keys=True, ensure_ascii=False)
                           for row in (json.loads(line) for line in (directory / 'cases.jsonl').read_text().splitlines())}
    for dataset in config['datasets']:
        queries = {json.dumps(row['messages'], sort_keys=True, ensure_ascii=False) for row in quality.load_dataset(dataset['path'])}
        if queries & calibration_queries:
            raise ValueError('校准集和评测集存在相同输入: ' + dataset['id'])
        assert_independent([json.loads(line) for line in (directory / 'cases.jsonl').read_text().splitlines()],
                           quality.load_dataset(dataset['path']))
    load_calibration(directory, digest(Path(config['source']) / 'tokenizer.json'))
    write_json(run_dir / 'calibration_manifest.json', {'path': str(directory), **manifest})
    return directory


def quantize(root, config, item, model_info, calibration, directory, environment, cancelled=None):
    recipe = RECIPES[item['recipe']]
    if recipe.backend == 'none':
        return Path(config['source'])
    if item.get('mode', 'fresh') == 'reuse':
        if not item.get('model'):
            raise ValueError('复用候选需要model路径')
        return Path(item['model'])
    if recipe.backend != 'llmcompressor':
        raise ValueError('新量化仅使用LLM Compressor；历史后端只能复用已有产物')
    available = models.memory_limit_gib()
    required = config.get('minimum_memory_gib', 32)
    if available is None or available < required:
        raise RuntimeError(f'当前配方要求至少{required}GiB内存，当前{available}GiB')
    backend = config['backends'][recipe.backend]
    environment = environment.copy()
    environment.update(backend.get('environment', {}))
    environment['OMP_NUM_THREADS'] = str(backend.get('cpu_threads', 2))
    output = directory / 'quantized'
    job = {'root': str(root), 'source': config['source'], 'output': str(output),
           'calibration': str(calibration), 'recipe': recipe.id, 'mtp_policy': config['mtp_policy'],
           'source_identity': model_info['identity']}
    write_json(directory / 'quantization_job.json', job)
    command = [backend['python'], str(root / 'scripts/quantize_compressor.py'), '--job', str(directory / 'quantization_job.json')]
    write_json(directory / 'quantization_command.json', {'command': command})
    run_logged(command, environment, directory / 'quantization.log', root, cancelled)
    if not (output / 'quantization_ready.json').is_file():
        raise ValueError('后端未提交量化导出完成标志')
    return output


def prepare_view(root, config, item, artifact, model_info, source_mtp, directory):
    identity, tensors = models.inspect_model(artifact, payload_hashes=True)
    recipe = RECIPES[item['recipe']]
    settings = copy.deepcopy(identity['config'])
    quant_config = settings.get('quantization_config', {})
    if recipe.backend == 'none':
        if quant_config:
            raise ValueError('原模型参照不能含量化配置')
    else:
        ready = Path(artifact) / 'quantization_ready.json'
        if not ready.is_file():
            raise ValueError('第一阶段复用候选需要量化来源记录；来源未知的模型可在第二阶段单独测试')
        provenance = read_json(ready)
        provenance_recipe = provenance.get('recipe')
        if isinstance(provenance_recipe, dict):
            from scripts.quantize_checkpoint import source_identity
            if provenance_recipe.get('source') != source_identity(Path(config['source'])):
                raise ValueError('量化候选并非来自当前原模型或来源已改变')
        elif (Path(provenance.get('source', '')).resolve() != Path(config['source']).resolve()
              or provenance.get('source_identity') != model_info['identity']):
            raise ValueError('量化候选来源或源权重指纹不匹配')
        method = quant_config.get('quant_method')
        if method != recipe.format:
            raise ValueError(f'期望{recipe.format}导出，实际{method}')
        if method in ('awq', 'gptq') and quant_config.get('bits') != (4 if recipe.weights == 'INT4' else 8):
            raise ValueError('导出位宽不匹配')
        if method == 'compressed-tensors':
            groups = list(quant_config.get('config_groups', {}).values())
            expected_bits = 4 if recipe.weights == 'INT4' else 8
            if not groups or any(group.get('weights', {}).get('num_bits') != expected_bits for group in groups):
                raise ValueError('compressed-tensors权重位宽与配方不符')
            expected_type = 'float' if recipe.weights == 'FP8' else 'int'
            if any(group.get('weights', {}).get('type') != expected_type for group in groups):
                raise ValueError('compressed-tensors权重数值类型与配方不符')
            for group in groups:
                activation = group.get('input_activations')
                if recipe.activations.startswith('FP8'):
                    if not activation or activation.get('num_bits') != 8 or activation.get('type') != 'float' or not activation.get('dynamic'):
                        raise ValueError('期望动态FP8激活，导出配置不符')
                elif recipe.activations.startswith('INT8'):
                    if not activation or activation.get('num_bits') != 8 or activation.get('type') != 'int' or not activation.get('dynamic'):
                        raise ValueError('期望动态INT8激活，导出配置不符')
                elif activation is not None:
                    raise ValueError('W4A16/W8A16不能隐式引入激活量化')
        packed = any(key.endswith(('.qweight', '.weight_packed')) for key in tensors)
        native_int8 = recipe.weights == 'INT8' and any(
            key.endswith('.weight') and value['dtype'] == 'I8' and key[:-7] + '.weight_scale' in tensors
            for key, value in tensors.items())
        native_fp8 = recipe.weights == 'FP8' and any(
            key.endswith('.weight') and value['dtype'].startswith('F8_') and key[:-7] + '.weight_scale' in tensors
            for key, value in tensors.items())
        if not (packed or native_int8 or native_fp8):
            raise ValueError('未发现真实压缩权重与缩放，不能将伪量化结果用于部署测试')
    actual_mtp = tensor_fingerprints(artifact, model_info['mtp_tensor_names'])
    if actual_mtp != source_mtp:
        raise ValueError('MTP权重发生变化或丢失；拒绝继续')
    for key in tensors:
        if re.search(r'(^|\.)mtp[._]', key) and key not in source_mtp:
            raise ValueError('发现未知或被量化的MTP张量: ' + key)
    view = directory / 'model'
    view.mkdir()
    for path in Path(artifact).iterdir():
        if path.is_file() and path.suffix in ('.json', '.jinja', '.txt', '.model') and path.stat().st_size < 100 * 1024**2:
            shutil.copy2(path, view / path.name)
    for name in identity['shards']:
        (view / name).symlink_to((Path(artifact) / name).resolve())
    if recipe.format in ('awq', 'gptq'):
        skipped = set(quant_config.get('modules_to_not_convert') or []) | {'lm_head', 'visual', 'mtp'}
        for name in model_info['protected_modules']:
            relative = name[name.index('layers.'):]
            skipped.update([relative, relative.rsplit('.', 1)[0] + '.in_proj_ba'])
        quant_config['modules_to_not_convert'] = sorted(skipped)
        if recipe.format == 'gptq':
            quant_config['modules_in_block_to_quantize'] = [sorted({name[:-8].split('layers.', 1)[-1].split('.', 1)[-1]
                for name in tensors if name.endswith('.qweight') and 'layers.' in name})]
    write_json(view / 'config.json', settings)
    receipt = {'artifact': str(artifact), 'load_view': str(view), 'identity': identity,
               'load_config_sha256': digest(view / 'config.json'),
               'recipe': item['recipe'], 'mtp_policy': config['mtp_policy'],
               'mtp_unchanged': actual_mtp == source_mtp, 'mtp_tensors': len(actual_mtp)}
    write_json(directory / 'model_manifest.json', receipt)
    return view


def inference_config(config, item):
    result = copy.deepcopy(config)
    result['audit'] = {}
    profile = copy.deepcopy(config.get('reference_profile', {'attention': 'fa2', 'graph': True, 'batched_tokens': 8192}))
    profile.update(id=item['id'], mtp=0)
    result['profiles'] = [profile]
    recipe = RECIPES[item['recipe']]
    result['server']['quantization'] = {'awq': 'awq_marlin', 'gptq': 'gptq_marlin'}.get(recipe.format)
    result['server']['worker_cls'] = 'pipeline.quantization_audit_worker.QuantizationAuditWorker'
    return result, profile



def run_candidate(root, config, item, preview, source_mtp, calibration, run_dir, cancelled, prepare_only=False):
    environment = runtime_environment(root, config)
    if cancelled.is_set():
        return
    directory = run_dir / 'candidates' / item['id']
    prior = read_json(directory / 'status.json') if (directory / 'status.json').exists() else {}
    if prior.get('status') in ('completed', 'skipped') or (prepare_only and prior.get('status') == 'prepared'):
        return
    prepared = prior.get('status') == 'prepared'
    if directory.exists() and not prepared:
        directory.rename(directory.with_name(directory.name + '.failed-' + str(time.time_ns())))
    directory.mkdir(parents=True, exist_ok=True)
    if not item['eligible']:
        write_json(directory / 'status.json', {'status': 'skipped', 'reason': item['reason']})
        print('[量化] 跳过 ' + item['id'] + '：' + item['reason'], flush=True)
        return
    print('[量化] ' + item['id'] + ('：准备模型' if not prepared else '：开始评测')
          + f'，GPU {config["runtime"]["gpu"]}', flush=True)
    try:
        runtime.ensure_idle(config)
        write_json(directory / 'progress.json', {'status': 'preparing' if not prepared else 'measuring',
                   'candidate': item['id'], 'pid': os.getpid(), 'gpu': config['runtime']['gpu']})
        if not prepared:
            write_json(directory / 'status.json', {'status': 'quantizing_or_loading'})
            artifact = quantize(root, config, item, preview['model'], calibration, directory, environment, cancelled)
            view = prepare_view(root, config, item, artifact, preview['model'], source_mtp, directory)
            assets = directory / 'audit'
            run_logged([config['runtime']['python'], str(root / 'scripts/prepare_qwen_assets.py'),
                        '--model', str(view), '--output', str(assets), '--audit-only'],
                       environment, directory / 'asset_preparation.log', root, cancelled)
            write_json(directory / 'status.json', {'status': 'prepared', 'quantization_gpu': config['runtime']['gpu']})
            print('[量化] ' + item['id'] + '：模型准备完成，等待统一条件评测。', flush=True)
        else:
            receipt = read_json(directory / 'model_manifest.json')
            actual, _ = models.inspect_model(receipt['artifact'], payload_hashes=True)
            if actual != receipt['identity'] or digest(Path(receipt['load_view']) / 'config.json') != receipt['load_config_sha256']:
                raise ValueError('已准备候选发生变化')
            view = Path(receipt['load_view'])
        if prepare_only:
            return
        execution, profile = inference_config(config, item)
        execution['audit'] = {'expected_bf16': str(directory / 'audit/expected_bf16.json')}
        write_json(directory / 'execution_config.json', execution)
        with runtime.server(root, execution, profile, view, directory, cancelled=cancelled) as (base, _):
            methods = read_json(directory / 'loading_audit.json').get('methods', {})
            if item['recipe'] == 'compressor_w8a8' and 'scheme:CompressedTensorsW8A8Int8' not in methods:
                raise RuntimeError('实际加载未确认W8A8 INT8计算，不能将回退结果纳入对比')
            for dataset in config['datasets']:
                output = directory / 'datasets' / dataset['id']
                output.mkdir(parents=True)
                rows = read_json(run_dir / 'datasets' / (dataset['id'] + '.json'))
                runtime.warmup(base, rows, config['benchmark'], output)
                write_json(directory / 'status.json', {'status': 'measuring', 'dataset': dataset['id'], 'gpu': config['runtime']['gpu']})
                if config['benchmark'].get('separate_quality'):
                    from .measurement import measure
                    measure(base, rows, config['benchmark'], output, item['id'])
                else:
                    benchmark.run_suite(base, rows, config['benchmark'], output, item['id'])
        write_json(directory / 'status.json', {'status': 'completed', 'finished_at': time.time(), 'evaluation_gpu': config['runtime']['gpu']})
        print('[量化] ' + item['id'] + '：评测完成。', flush=True)
    except Exception as error:
        write_json(directory / 'status.json', {'status': 'failed', 'error': str(error)})
        print(json.dumps({'candidate': item['id'], 'error': str(error)}, ensure_ascii=False), flush=True)
        if item['recipe'] == 'original':
            raise RuntimeError('原模型参照失败，不能进行量化质量对照') from error


def run(root, config_path, run_id, resume=False):
    root = Path(root)
    config = load_config(root, config_path)
    preview = plan(root, config)
    source_identity, _ = models.inspect_model(config['source'], payload_hashes=True)
    run_dir = inside(root, root / 'runs' / identifier(run_id))
    signature_data = {'config': config, 'code': source_digest(root), 'source': source_identity,
                      'environment': environment_fingerprints(root, config),
                      'data': {item['id']: digest(item['path']) for item in config['datasets']}}
    calibration_settings = config.get('calibration', {})
    if calibration_settings.get('mode') == 'frozen':
        calibration_path = (root / calibration_settings['path']).resolve()
        signature_data['calibration'] = {name: digest(calibration_path / name)
                                         for name in ['manifest.json', 'cases.jsonl', 'tokens.jsonl']}
    if calibration_settings.get('mode') == 'official' and calibration_settings.get('raw_data'):
        raw_data = (root / calibration_settings['raw_data']).resolve()
        signature_data['official_calibration_source'] = {
            'records_sha256': digest(raw_data), 'manifest_sha256': digest(raw_data.with_suffix('.manifest.json'))}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    if resume:
        if read_json(run_dir / 'manifest.json')['signature'] != signature:
            raise ValueError('代码、配置、来源或数据改变，请新建run-id')
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / 'manifest.json', {'signature': signature, 'started_at': time.time(), 'inputs': signature_data})
        write_json(run_dir / 'config.json', config)
        for relative in source_digest(root):
            target = run_dir / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, target)
        for dataset in config['datasets']:
            from .modes import sample_rows
            write_json(run_dir / 'datasets' / (dataset['id'] + '.json'),
                       sample_rows(quality.load_dataset(dataset['path']), config['benchmark'].get('sample_limit')))
    from .quantization_report import generate
    from .text_report import quantization_summary, report_paths
    def publish_candidate(candidate):
        result = generate(run_dir)
        print(quantization_summary(result, candidate), flush=True)
        report_paths(run_dir)
    try:
        with ExitStack() as leases:
            for gpu in config['runtime'].get('quantization_gpus', [config['runtime'].get('gpu', 0)]):
                gpu_config = copy.deepcopy(config)
                gpu_config['runtime']['gpu'] = gpu
                leases.enter_context(runtime.gpu_lock(root, gpu_config))
            probe, _ = inference_config(config, config['candidates'][0])
            doctor = runtime.doctor(root, probe, run_dir)
            if resume and (run_dir / 'gpu_identity.json').exists() and read_json(run_dir / 'gpu_identity.json')['uuid'] != doctor['gpu_uuid']:
                raise ValueError('GPU改变，拒绝混合对照结果')
            write_json(run_dir / 'gpu_identity.json', {'uuid': doctor['gpu_uuid']})
            preview = plan(root, config, doctor['capability'])
            preview['model']['identity'] = source_identity
            write_json(run_dir / 'plan.json', preview)
            source_mtp = tensor_fingerprints(config['source'], preview['model']['mtp_tensor_names'])
            write_json(run_dir / 'source_mtp.json', source_mtp)
            environment = runtime_environment(root, config)
            calibration = None
            if (config.get('calibration', {}).get('mode') == 'frozen' or any(
                    item['eligible'] and item.get('mode', 'fresh') == 'fresh' and item['recipe'] != 'original'
                    for item in preview['candidates'])):
                if resume and (run_dir / 'calibration_manifest.json').exists():
                    calibration = Path(read_json(run_dir / 'calibration_manifest.json')['path'])
                else:
                    calibration = prepare_calibration(root, config, run_dir, environment)
            cancelled = threading.Event()
            run_candidate(root, config, preview['candidates'][0], preview, source_mtp, calibration, run_dir, cancelled)
            publish_candidate(preview['candidates'][0]['id'])
            remaining = preview['candidates'][1:]
            quantization_gpus = config['runtime'].get('quantization_gpus', [config['runtime'].get('gpu', 0)])
            prepared_count = 0
            progress_lock = threading.Lock()
            def lane(index, gpu):
                nonlocal prepared_count
                lane_config = copy.deepcopy(config)
                lane_config['runtime']['gpu'] = gpu
                lane_config['server']['port'] = config['server']['port'] + index
                for item in remaining[index::len(quantization_gpus)]:
                    if cancelled.is_set():
                        return
                    run_candidate(root, lane_config, item, preview, source_mtp, calibration, run_dir, cancelled, prepare_only=True)
                    if cancelled.is_set():
                        return
                    with progress_lock:
                        prepared_count += 1
                        state = read_json(run_dir / 'candidates' / item['id'] / 'status.json').get('status')
                        label = {'prepared': '已准备', 'completed': '已完成', 'skipped': '已跳过', 'failed': '失败'}.get(state, state)
                        preparation_progress.update(prepared_count, detail=item['id'] + '：' + str(label), force=True)
            write_json(run_dir / 'status.json', {'status': 'running_candidates', 'gpus': quantization_gpus, 'pid': os.getpid()})
            with Progress('量化候选已处理（含跳过/失败）', len(remaining), unit='候选', live=False) as preparation_progress, \
                    ThreadPoolExecutor(max_workers=len(quantization_gpus)) as executor:
                futures = [executor.submit(lane, index, gpu) for index, gpu in enumerate(quantization_gpus)]
                try:
                    for future in futures:
                        future.result()
                except BaseException:
                    cancelled.set()
                    raise
            for item in remaining:
                status_path = run_dir / 'candidates' / item['id'] / 'status.json'
                if status_path.is_file() and read_json(status_path).get('status') == 'prepared':
                    run_candidate(root, config, item, preview, source_mtp, calibration, run_dir, cancelled)
                publish_candidate(item['id'])
            current_source, _ = models.inspect_model(config['source'], payload_hashes=True)
            if current_source != source_identity:
                raise ValueError('执行期间原模型发生变化，拒绝提交对照报告')
            result = generate(run_dir)
            failed = any(item['status'] == 'failed' for item in result['statuses'].values())
            write_json(run_dir / 'status.json', {'status': 'completed_with_failures' if failed else 'completed',
                                                'report': str(run_dir / 'report.html'), 'finished_at': time.time()})
            print(quantization_summary(result), flush=True)
            report_paths(run_dir)
            return 1 if failed else 0
    except BaseException as error:
        write_json(run_dir / 'status.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', 'error': str(error)})
        raise


def select(root, run_id, candidate_id, save=True):
    from .quantization_report import generate
    run_dir = inside(root, Path(root) / 'runs' / identifier(run_id))
    report = generate(run_dir) if save else generate(run_dir, persist=False)
    requested_candidate = candidate_id
    if candidate_id == 'recommended':
        candidate_id = report['recommendation']['candidate']
    if not candidate_id or not report['selectable'].get(candidate_id):
        raise ValueError('只能选择已完成本次完整测试的候选；推荐质量条件不限制手动选择')
    directory = run_dir / 'candidates' / identifier(candidate_id)
    receipt = read_json(directory / 'model_manifest.json')
    if receipt.get('mtp_unchanged') is not True or read_json(directory / 'loading_audit.json').get('passed') is not True:
        raise ValueError('候选未通过MTP完整性或实际加载检查')
    actual, _ = models.inspect_model(receipt['artifact'], payload_hashes=True)
    if actual != receipt['identity']:
        raise ValueError('实测模型已改变')
    if digest(Path(receipt['load_view']) / 'config.json') != receipt['load_config_sha256']:
        raise ValueError('实测加载配置已改变')
    config = read_json(run_dir / 'config.json')
    entry = next(item for item in config['candidates'] if item['id'] == candidate_id)
    calibration_manifest = run_dir / 'calibration_manifest.json'
    handoff = {'schema_version': 1, 'stage': 'quantization_selection', 'candidate': candidate_id,
               'experience': config.get('experience', {}),
               'model': receipt['load_view'], 'artifact': receipt['artifact'], 'identity': receipt['identity'],
               'mtp_policy': config['mtp_policy'], 'mtp_source': config.get('mtp_source', config['source']),
               'quality_gate': report['candidate_gates'][candidate_id], 'report': str(run_dir / 'report.html'),
               'selection': {'mode': 'user', 'requested_candidate': requested_candidate,
                             'recommended_candidate': report['recommendation']['candidate'],
                             'quality_criteria_are_advisory': True},
               'quantization': {'recipe': entry['recipe'], 'backend': RECIPES[entry['recipe']].backend},
               'calibration': read_json(calibration_manifest) if calibration_manifest.is_file() else None,
               'acceleration_tested': False, 'production_validated': False}
    if not save:
        return handoff
    write_json(run_dir / 'selected_model.json', handoff)
    print(json.dumps({'selection': str(run_dir / 'selected_model.json'), 'candidate': candidate_id}, ensure_ascii=False))
    import shlex
    next_run = run_id[:80] + '-acceleration'
    print('第二阶段入口：' + shlex.join([str(Path(root) / 'quickstart'), 'accelerate', 'run',
                                     '--model', str(run_dir / 'selected_model.json'), '--run-id', next_run]))
    return handoff


def interactive(root):
    default = Path(root) / 'configs/quantization-llmcompressor.local.json'
    if not default.is_file():
        default = Path(root) / 'configs/quantization.example.json'
    answer = input(f'量化配置路径 [{default}]: ').strip()
    path = Path(answer) if answer else default
    config = load_config(root, path)
    source = input(f'完整精度模型目录 [{config["source"]}]: ').strip() or config['source']
    changed = Path(source).resolve() != Path(config['source']).resolve()
    config['source'] = str(Path(source).resolve())
    model = inspect_source(Path(config['source']))
    options = [name for name in FRESH_RECIPES if eligibility(name, model)['eligible']]
    print('模型适配器：' + model['adapter']['family'])
    print('量化配方：' + ', '.join(options) + '；后端未配置或GPU不支持的配方会明确跳过。')
    default_recipes = ','.join(item['recipe'] for item in config['candidates'] if item['recipe'] in options) or 'all'
    requested = input(f'选择配方，逗号分隔；all=全部兼容配方 [{default_recipes}]: ').strip() or default_recipes
    names = options if requested.lower() == 'all' else list(dict.fromkeys(name.strip() for name in requested.split(',') if name.strip()))
    if not names or any(name not in options for name in names):
        raise ValueError('请选择列表中的量化配方')
    prior = {item['recipe']: item for item in config['candidates']}
    config['candidates'] = [{'id': 'original', 'recipe': 'original', 'mode': 'reuse'}]
    for name in names:
        config['candidates'].append(prior[name] if name in prior and not changed else {'id': name, 'recipe': name, 'mode': 'fresh'})
    custom = input('自定义评测JSON/JSONL路径，回车使用默认带答案100条: ').strip()
    config['datasets'] = [config['datasets'][0]]
    if custom:
        quality.load_dataset(custom)
        config['datasets'].append({'id': 'custom', 'path': str(Path(custom).resolve())})
    if changed:
        config['mtp_source'] = config['source']
        config['calibration'] = official_defaults()
    run_id = 'quantization-' + time.strftime('%Y%m%d-%H%M%S')
    output = Path(root) / 'configs' / (run_id + '.local.json')
    write_json(output, config)
    result = run(root, output, run_id)
    if result == 0:
        chosen = input('输入要交给第二阶段的候选ID（或recommended），回车只保留报告: ').strip()
        if chosen:
            select(root, run_id, chosen)
    return result
