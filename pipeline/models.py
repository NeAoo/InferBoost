"""Read-only source verification and isolated runtime loading views."""
import copy
import json
from pathlib import Path
import shutil
import struct
import subprocess
from .common import digest, read_json, write_json, inside


def memory_limit_gib():
    limits=[]
    for name in ['/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes']:
        path=Path(name)
        if path.is_file():
            value=path.read_text().strip()
            if value.isdigit() and int(value)<2**60:limits.append(int(value))
    path=Path('/proc/meminfo')
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith('MemTotal:'):limits.append(int(line.split()[1])*1024)
    return min(limits)/1024**3 if limits else None


def inspect_model(model, payload_hashes=False):
    model = Path(model)
    config = read_json(model / 'config.json')
    index = model / 'model.safetensors.index.json'
    mapping = read_json(index)['weight_map'] if index.exists() else None
    names = sorted(set(mapping.values())) if mapping else ['model.safetensors']
    shards, tensors = {}, {}
    for name in names:
        if Path(name).name != name or not name.endswith('.safetensors'):
            raise ValueError(f'非法模型分片名称: {name}')
        path = model / name
        with path.open('rb') as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f'残缺分片: {path}')
            length = struct.unpack('<Q', raw)[0]
            if not 2 <= length <= 64 * 1024**2:
                raise ValueError('safetensors 文件头异常')
            header = json.loads(stream.read(length))
        end = 0
        entries = [(key, value) for key, value in header.items() if key != '__metadata__']
        for key, value in sorted(entries, key=lambda item: item[1]['data_offsets'][0]):
            begin, finish = value['data_offsets']
            if begin != end or finish < begin or key in tensors or (mapping is not None and mapping.get(key) != name):
                raise ValueError('分片偏移、重复张量或索引异常')
            tensors[key] = value
            end = finish
        if not entries or 8 + length + end != path.stat().st_size:
            raise ValueError(f'分片长度异常: {path}')
        shards[name] = {'bytes': path.stat().st_size, 'resolved_path': str(path.resolve()),
                        'sha256': digest(path) if payload_hashes else None}
    if mapping is not None and set(mapping) != set(tensors):
        raise ValueError('分片缺少索引声明的张量')
    return {'config': config, 'config_sha256': digest(model / 'config.json'),
            'index_sha256': digest(index) if index.exists() else None,
            'shards': shards, 'tensor_count': len(tensors)}, tensors


def prepare(root, config, run_dir, environment):
    quantization = config['quantization']
    if quantization['mode'] == 'fresh':
        available=memory_limit_gib()
        required=quantization.get('minimum_memory_gib',400)
        if available is None or available<required:
            raise RuntimeError(f'该量化配方要求至少{required}GiB内存；当前上限{available}GiB。不能只靠减少编译并发解决激活缓存峰值。')
        source = Path(quantization['source'])
        if not (source / 'config.json').is_file():
            raise FileNotFoundError(f'完整精度模型不存在，不能开始重新量化: {source}')
        source_type=read_json(source/'config.json').get('model_type')
        if source_type not in ('qwen3_5','qwen3_5_moe'):
            raise ValueError('本版量化适配器仅验证Qwen3.5 dense/MoE架构')
        output = inside(root, run_dir / 'quantized' / 'awq4')
        command = [quantization['python'], str(Path(root)/'scripts/quantize_checkpoint.py'),
                   '--root', str(root), '--source', str(source), '--output', str(output),
                   '--calibration', quantization['calibration'], '--bits', '4', '--method', 'awq',
                   '--awq-qwen-dense-norm-fix' if source_type=='qwen3_5' else '--awq-qwen-norm-fix']
        if quantization.get('activation_cache')=='disk':
            command+=['--awq-disk-cache','--disk-cache-gib',str(quantization.get('disk_cache_gib',500)),
                      '--rss-limit-gib',str(quantization.get('rss_limit_gib',180))]
        elif source_type=='qwen3_5_moe':
            command+=['--awq-deduplicate-cache']
        quant_environment = environment.copy()
        quant_environment.update(quantization.get('environment', {}))
        quant_environment['TMPDIR'] = str(Path(root) / 'tmp')
        # Quantization uses its own CUDA/Python environment, not the inference runtime.
        if not quant_environment.get('GPTQMODEL_AWQ_BUILD_ROOT'):
            raise ValueError('新量化必须配置独立的 GPTQMODEL_AWQ_BUILD_ROOT 和已核验内核')
        write_json(run_dir / 'quantization_command.json', {'command': command})
        with (run_dir / 'quantization.log').open('a') as log:
            subprocess.run([quantization['python'],str(Path(root)/'scripts/prepare_awq_kernel.py'),'--build','--require-gpu'],
                           env=quant_environment,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run(command, env=quant_environment, cwd=root, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        model = output
    else:
        model = Path(quantization['model'])
    identity, tensors = inspect_model(model, payload_hashes=True)
    quant = identity['config'].get('quantization_config', {})
    if quant.get('quant_method') != 'awq' or quant.get('bits') != 4:
        raise ValueError('要求 AWQ4 权重，不能替换为其他算法')
    if quant.get('group_size') != 128 or not quant.get('zero_point'):
        raise ValueError('本版要求 group_size=128 和 zero_point=true')
    ready = model / 'quantization_ready.json'
    provenance = read_json(ready) if ready.is_file() else None
    if provenance and provenance.get('shards'):
        for name, expected in provenance['shards'].items():
            if identity['shards'].get(name, {}).get('sha256') != expected:
                raise ValueError('权重 SHA256 与原导出清单不符')
    view = inside(root, run_dir / 'model')
    view.mkdir(exist_ok=False)
    # Copy small configuration files; keep immutable weight files as read-only references.
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                 'generation_config.json', 'special_tokens_map.json', 'chat_template.jinja',
                 'model.safetensors.index.json', 'vocab.json', 'merges.txt']:
        if (model / name).is_file():
            shutil.copy2(model / name, view / name)
    for name in identity['shards']:
        (view / name).symlink_to((model / name).resolve())
    settings = copy.deepcopy(identity['config'])
    quant = settings['quantization_config']
    protected = []
    if settings.get('model_type') in ('qwen3_5_moe','qwen3_5'):
        skipped = set(quant.get('modules_to_not_convert') or []) | {'visual', 'lm_head'}
        for key, tensor in tensors.items():
            if '.linear_attn.in_proj_' in key and key.endswith('.weight'):
                if tensor['dtype'] != 'BF16':
                    raise ValueError(f'预期保护的投影并非BF16: {key}')
                module = key[: -len('.weight')]
                module = module[module.index('layers.'):]
                skipped.add(module)
                if module.endswith(('.in_proj_a', '.in_proj_b')):
                    skipped.add(module.rsplit('.', 1)[0] + '.in_proj_ba')
                protected.append(module)
        quant['modules_to_not_convert'] = sorted(skipped)
        write_json(view / 'config.json', settings)
    receipt = {'mode': 'reused' if quantization['mode'] == 'reuse' else 'fresh_export',
               'source_model': str(model), 'load_view': str(view),
               'identity': identity, 'quantization_provenance': provenance,
               'protected_modules': protected, 'quality_validated_this_run': False}
    if config.get('auto_qwen_assets'):
        assets=run_dir/'assets'
        subprocess.run([config['runtime']['python'],str(Path(root)/'scripts/prepare_qwen_assets.py'),
            '--model',str(view),'--mtp-source',config['mtp_source'],'--output',str(assets)],
            env=environment,cwd=root,check=True)
        config['audit']={'expected_bf16':str(assets/'expected_bf16.json'),
                         'mtp_reference':str(assets/'mtp_reference.json')}
        config['mtp_model']=str(assets/'draft')
    write_json(run_dir/'execution_config.json',config)
    # Commit readiness only after all dependent assets and execution settings exist.
    write_json(run_dir / 'model_manifest.json', receipt)
    return view
