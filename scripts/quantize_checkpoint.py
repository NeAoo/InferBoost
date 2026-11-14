"""GPTQModel制作GPTQ 8/4bit与AWQ 4bit；冻结校准token，不自动截断长上下文。"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time
from contextlib import ExitStack
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(path):
    config = json.loads((path / 'config.json').read_text())
    if config.get('quantization_config'):
        raise ValueError('输入必须是未量化的原checkpoint')
    index = path / 'model.safetensors.index.json'
    names = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.is_file() else ['model.safetensors']
    return {'path': str(path), 'config_sha256': file_hash(path / 'config.json'),
            'index_sha256': file_hash(index) if index.is_file() else None,
            'tokenizer_sha256': file_hash(path / 'tokenizer.json'),
            'weight_files': {name: {'bytes': (path / name).stat().st_size,
                                   'mtime_ns': (path / name).stat().st_mtime_ns} for name in names}}


CALIBRATION_VERSION = 'task-stratified-token-windows-v2'


def validate_export(path, bits, method='gptq', *, normalize_config=True):
    config = json.loads((path / 'config.json').read_text())
    quant = config.get('quantization_config', {})
    if quant.get('quant_method', quant.get('method')) != method or quant.get('bits') != bits:
        raise ValueError('导出权重不是要求的量化算法或位宽')
    if quant.get('group_size') != 128:
        raise ValueError('导出量化分组参数与实验设计不一致')
    if method == 'gptq':
        if quant.get('desc_act') is not False or quant.get('sym') is not True:
            raise ValueError('导出GPTQ参数与实验设计不一致')
        if quant.get('checkpoint_format', quant.get('format', 'gptq')) != 'gptq':
            raise ValueError('只接受标准GPTQ布局')
        quant.update(quant_method='gptq', checkpoint_format='gptq')
    elif method == 'awq':
        if bits != 4 or quant.get('sym') is True or quant.get('zero_point') is not True:
            raise ValueError('AWQ必须为4bit非对称量化，保留zero_point')
        if str(quant.get('version', quant.get('format', ''))).lower() != 'gemm':
            raise ValueError('只接受vLLM可读的AWQ GEMM布局')
        quant.update(quant_method='awq', version='gemm', zero_point=True)
        # 与制作时的显式排除一致；MoE WNA16不自动从权重dtype发现未量化视觉层。
        skipped = quant.get('modules_to_not_convert') or []
        if not isinstance(skipped,list) or not all(isinstance(name,str) for name in skipped):
            raise ValueError('AWQ未量化模块清单不合法')
        quant['modules_to_not_convert'] = sorted(set(skipped + ['visual','lm_head']))
    else:
        raise ValueError('未知量化算法')
    # 只规范化新导出文件的等价字段，不写原始checkpoint。
    if normalize_config:
        (path / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    index = path / 'model.safetensors.index.json'
    names = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.is_file() else ['model.safetensors']
    if not names or any(not (path / name).is_file() for name in names):
        raise ValueError('量化分片不完整')
    return quant, names


def quantization_recipe(identity, calibration, bits, method, *, awq_qwen_norm_fix=False,awq_qwen_dense_norm_fix=False):
    recipe = {'source': identity, 'calibration_sha256': calibration['cases_sha256'],
            'calibration_tokens_sha256': calibration['tokens_sha256'],
            'bits': bits, 'group_size': 128, 'desc_act': False, 'sym': method == 'gptq',
            'algorithm': method.upper(), 'tool_version': '7.5.0',
            'protected_policy': 'mtp_visual_lmhead_gdn_ab_original_precision_v1'}
    if awq_qwen_norm_fix:
        if method != 'awq' or bits != 4:
            raise ValueError('Qwen norm 修复仅用于 AWQ 4bit')
        from scripts.awq_qwen_norm import fix_metadata
        recipe['awq_compatibility_fix'] = fix_metadata()
    if awq_qwen_dense_norm_fix:
        from scripts.awq_qwen_dense_norm import fix_metadata
        recipe['awq_compatibility_fix']=fix_metadata()
    return recipe


def verify_reusable_export(path, expected):
    """只读检查指定成品；缺失、损坏或配方不符就停止，绝不改为重新量化。"""
    path = path.resolve()
    ready = json.loads((path / 'quantization_ready.json').read_text())
    if ready.get('recipe') != expected:
        raise ValueError('复用权重来源或量化配方不一致；不会重新量化')
    _, names = validate_export(path, expected['bits'], expected['algorithm'].lower(), normalize_config=False)
    index_path = path / 'model.safetensors.index.json'
    weight_map = json.loads(index_path.read_text())['weight_map'] if index_path.exists() else None
    known_hashes = ready.get('shards', {})
    if known_hashes and set(known_hashes) != set(names):
        raise ValueError('复用权重校验清单与分片不一致')
    seen, total_bytes = set(), 0
    for name in names:
        if Path(name).name != name or not name.endswith('.safetensors'):
            raise ValueError('复用权重分片路径不合法')
        shard = path / name
        if shard.is_symlink():
            raise ValueError('复用分片不能是符号链接')
        with shard.open('rb') as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError('复用分片缺少文件头')
            header_bytes = struct.unpack('<Q', raw)[0]
            if not 2 <= header_bytes <= 64 * 1024**2:
                raise ValueError('复用分片文件头长度不合法')
            header = json.loads(stream.read(header_bytes))
            tensors = {k: v for k, v in header.items() if k != '__metadata__'}
            end = 0
            for tensor, item in sorted(tensors.items(), key=lambda row: row[1]['data_offsets'][0]):
                start, stop = item['data_offsets']
                if start != end or stop < start or tensor in seen:
                    raise ValueError('复用分片包含无效偏移或重复张量')
                if weight_map is not None and weight_map.get(tensor) != name:
                    raise ValueError('复用分片与权重索引不一致')
                seen.add(tensor)
                end = stop
            if not tensors or 8 + header_bytes + end != shard.stat().st_size:
                raise ValueError('复用分片不完整，禁止部署残缺权重')
            if name in known_hashes:
                stream.seek(0)
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(8 * 1024**2), b''):
                    digest.update(block)
                if digest.hexdigest() != known_hashes[name]:
                    raise ValueError('复用分片SHA256不匹配')
        total_bytes += shard.stat().st_size
    if weight_map is not None and seen != set(weight_map):
        raise ValueError('复用权重存在缺失张量')
    return {'path': str(path), 'mode': 'read_only_reuse', 'shards': len(names),
            'tensors': len(seen), 'weight_bytes': total_bytes,
            'payload_hashes_verified': bool(known_hashes), 'quantization_will_run': False}


def load_calibration(path, tokenizer_sha256):
    """所有算法读取完全相同的冻结token；损坏或配方不符时失败，不默默重切。"""
    audit = json.loads((path/'manifest.json').read_text())
    if audit.get('version')=='pipeline-synthetic-calibration-v1':
        if audit.get('tokenizer_sha256')!=tokenizer_sha256 or audit.get('test_query_overlap')!=0:
            raise ValueError('校准 tokenizer 或测试隔离不符')
        if audit['cases_sha256']!=file_hash(path/'cases.jsonl') or audit['tokens_sha256']!=file_hash(path/'tokens.jsonl'):
            raise ValueError('校准文件SHA256不符')
        rows=[json.loads(line) for line in (path/'tokens.jsonl').read_text().splitlines()]
        if len(rows)!=audit['count'] or len({row['id'] for row in rows})!=len(rows):
            raise ValueError('校准样本数量或ID不符')
        for row in rows:
            ids=row['input_ids']
            if not 128<=len(ids)<=audit['max_tokens'] or not all(type(token) is int and token>=0 for token in ids):
                raise ValueError('校准token非法')
        return audit,[{'input_ids':row['input_ids'],'attention_mask':[1]*len(row['input_ids'])} for row in rows]
    if (audit.get('version') != CALIBRATION_VERSION or audit.get('test_query_overlap') != 0
            or audit.get('tokenizer_sha256') != tokenizer_sha256):
        raise ValueError('校准版本、评测隔离或tokenizer指纹不符，请先离线构建新版校准集')
    if audit['cases_sha256'] != file_hash(path/'cases.jsonl') or audit['tokens_sha256'] != file_hash(path/'tokens.jsonl'):
        raise ValueError('校准文件SHA256不一致')
    cases = [json.loads(line) for line in (path/'cases.jsonl').read_text().splitlines()]
    tokens = [json.loads(line) for line in (path/'tokens.jsonl').read_text().splitlines()]
    if len(cases) != len(tokens) or len(cases) != audit['count']:
        raise ValueError('校准记录数量与manifest不一致')
    calibration, seen = [], set()
    for case, row in zip(cases, tokens):
        ids = row.get('input_ids')
        if (case.get('id') != row.get('id') or row['id'] in seen or not case.get('target_complete')
                or not isinstance(ids, list) or not 128 <= len(ids) <= 65536
                or len(ids) != case.get('selected_tokens')
                or not all(type(i) is int and i >= 0 for i in ids)):
            raise ValueError('校准token损坏、目标不完整或长度不合法')
        seen.add(row['id'])
        calibration.append({'input_ids': ids, 'attention_mask': [1]*len(ids)})
    kinds = Counter(case.get('kind') for case in cases)
    if (dict(kinds) != audit.get('kinds') or not all(kinds.get(k) for k in ['tool','summary','long_summary'])
            or any(c['kind']=='long_summary' and (not c.get('complete_context') or c['selected_tokens'] < 32768) for c in cases)):
        raise ValueError('校准集缺少工具、总结或完整长上下文，或覆盖统计不一致')
    return audit, calibration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--bits', type=int, choices=[4, 8], required=True)
    parser.add_argument('--method', choices=['gptq','awq'], default='gptq')
    parser.add_argument('--awq-deduplicate-cache', action='store_true',
                        help='逐字节相同的专家gate/up缓存共享存储；校准规则保持不变')
    parser.add_argument('--awq-qwen-norm-fix', action='store_true',
                        help='隔离修复Qwen3.5 MoE的offset RMSNorm；禁止复用旧错误权重')
    parser.add_argument('--awq-qwen-dense-norm-fix',action='store_true')
    parser.add_argument('--awq-disk-cache', action='store_true',
                        help='MoE激活保存到磁盘，保持原有token行选择规则')
    parser.add_argument('--disk-cache-gib',type=int,default=500)
    parser.add_argument('--rss-limit-gib',type=int,default=180)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    root, source, output = args.root.resolve(), args.source.resolve(), args.output.resolve()
    if root not in output.parents or output == source or source in output.parents:
        raise ValueError('量化输出必须为项目内独立目录，不得写原始权重')
    if args.method == 'awq' and args.bits != 4:
        raise ValueError('本轮AWQ对照仅支持4bit')
    if args.awq_deduplicate_cache and args.method != 'awq':
        raise ValueError('缓存去重只适用于AWQ，禁止修改GPTQ路径')
    if args.awq_disk_cache and (args.method!='awq' or args.awq_deduplicate_cache):
        raise ValueError('磁盘缓存只用于AWQ，且不能叠加旧内存去重')
    identity = source_identity(source)
    audit, calibration = load_calibration(args.calibration, identity['tokenizer_sha256'])
    expected = quantization_recipe(identity, audit, args.bits, args.method,
                                   awq_qwen_norm_fix=args.awq_qwen_norm_fix,
                                   awq_qwen_dense_norm_fix=args.awq_qwen_dense_norm_fix)
    if args.awq_qwen_dense_norm_fix and (args.awq_qwen_norm_fix or args.method!='awq' or json.loads((source/'config.json').read_text()).get('model_type')!='qwen3_5'):
        raise ValueError('Dense norm修复只允许用于Qwen3.5 dense AWQ')
    if args.awq_qwen_norm_fix:
        if json.loads((source / 'config.json').read_text()).get('model_type') != 'qwen3_5_moe':
            raise ValueError('该AWQ norm修复仅为已核验的Qwen3.5 MoE架构准备')
    optimization = None
    if args.awq_deduplicate_cache:
        from scripts.awq_activation_cache import optimization_metadata
        optimization = optimization_metadata()
    if args.awq_disk_cache:
        from scripts.awq_disk_cache import PATCH_VERSION
        optimization={'version':PATCH_VERSION,'calibration_changed':False,
                      'sampling_rule_changed':False,'max_disk_gib':args.disk_cache_gib,
                      'rss_limit_gib':args.rss_limit_gib,
                      'patch_sha256':file_hash(ROOT/'scripts/awq_disk_cache.py')}
    ready = output / 'quantization_ready.json'
    if ready.is_file():
        prior = json.loads(ready.read_text())
        if prior['recipe'] != expected:
            raise ValueError('已有量化产物来自不同checkpoint或配方，请用新的输出目录')
        validate_export(output, args.bits, args.method)
        print(f'[quantization reuse] {output}', flush=True)
        return 0
    if output.exists():
        raise ValueError('量化目录已存在但未完成；请核查中断日志后指定新目录，不覆盖部分权重')
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'output': str(output), 'recipe': expected,
                          'runtime_optimization': optimization}, ensure_ascii=False))
        return 0
    import torch
    from importlib.metadata import version
    from transformers import AutoTokenizer
    from gptqmodel import AWQConfig, GPTQConfig, GPTQModel
    if version('gptqmodel') != expected['tool_version'] or not torch.cuda.is_available():
        raise ValueError('需要独立GPTQModel 7.5.0环境及可用CUDA GPU')
    if torch.cuda.device_count() != 1:
        raise ValueError('量化子进程必须通过CUDA_VISIBLE_DEVICES只看到指定的一张卡')
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    if args.awq_deduplicate_cache:
        from scripts.awq_activation_cache import verify_processor
        verify_processor()
    if args.awq_qwen_norm_fix:
        from scripts.awq_qwen_norm import verify_environment
        verify_environment()
    if args.awq_qwen_dense_norm_fix:
        from scripts.awq_qwen_dense_norm import verify_environment
        verify_environment()
    output.mkdir(parents=True)
    started = time.monotonic()
    (output / 'quantization_status.json').write_text(json.dumps({'status':'quantizing','recipe':expected,
        'runtime_optimization':optimization},ensure_ascii=False,indent=2))
    config_class = GPTQConfig if args.method == 'gptq' else AWQConfig
    config = config_class(bits=args.bits, group_size=128, desc_act=False, sym=args.method == 'gptq',
        format='gptq' if args.method == 'gptq' else 'gemm', device='cuda:0', offload_to_disk=True,
        offload_to_disk_path=str(output / 'offload'),
        dynamic={r'-:.*visual.*': {}, r'-:.*lm_head.*': {}, r'-:.*mtp.*': {},
                 r'-:.*linear_attn.in_proj_[ab]$': {}})
    # 7.5.0 会把额外 kwargs 传给 meta 模型构造器；local_files_only 只适用于
    # HF 文件加载，不能传到这里。本地绝对路径及子进程的 HF_HUB_OFFLINE 保持离线。
    cache_statistics = None
    disk_store = None
    norm_statistics = None
    with ExitStack() as stack:
        from scripts.awq_activation_cache import memory_monitor
        stack.enter_context(memory_monitor(output))
        if args.awq_qwen_norm_fix:
            from scripts.awq_qwen_norm import corrected_qwen_norm_scaling
            norm_statistics = stack.enter_context(corrected_qwen_norm_scaling(output / 'norm_scaling.jsonl'))
            print('[awq qwen norm fix] ' + json.dumps(expected['awq_compatibility_fix']), flush=True)
        if args.awq_qwen_dense_norm_fix:
            from scripts.awq_qwen_dense_norm import corrected_dense_norm_scaling
            norm_statistics=stack.enter_context(corrected_dense_norm_scaling())
        if args.awq_deduplicate_cache:
            from scripts.awq_activation_cache import exact_deduplicated_capture, memory_monitor
            cache_statistics = stack.enter_context(exact_deduplicated_capture(output / 'cache_dedup.jsonl'))
            print('[awq exact cache dedup] ' + json.dumps(optimization), flush=True)
        if args.awq_disk_cache:
            from scripts.awq_disk_cache import disk_backed_capture
            from scripts.awq_activation_cache import memory_monitor
            disk_store=stack.enter_context(disk_backed_capture(output/'activation_spill',output/'disk_cache.jsonl',
                max_disk_gib=args.disk_cache_gib,rss_limit_gib=args.rss_limit_gib))
            print('[awq disk cache] '+json.dumps(optimization),flush=True)
        model = GPTQModel.load(str(source), config, trust_remote_code=False)
        model.quantize(calibration, batch_size=1)
        if args.awq_qwen_norm_fix and (not norm_statistics['corrected_calls']
                or not norm_statistics['compensated_moe_groups']):
            raise ValueError('AWQ norm或MoE路由补偿未实际生效，拒绝导出')
        if args.awq_qwen_dense_norm_fix and (not norm_statistics['corrected_calls'] or not norm_statistics['compensated_gdn_groups']):
            raise ValueError('Dense norm修复未生效')
        model.save(str(output))
    tokenizer.save_pretrained(output)
    quant, names = validate_export(output, args.bits, args.method)
    if source_identity(source) != expected['source']:
        raise ValueError('制作过程中原checkpoint发生变化，拒绝标记完成')
    status = {'status':'exported_not_serving_validated','recipe':expected,
              'runtime_optimization':optimization,'cache_statistics':disk_store.stats() if disk_store else cache_statistics,
              'norm_scaling_statistics':norm_statistics,
              'elapsed_seconds':round(time.monotonic()-started,2),'samples':len(calibration),
              'weight_bytes':sum((output/name).stat().st_size for name in names),
              'quantization_config':quant,'versions':{k:version(k) for k in ['gptqmodel','torch','transformers']},
              'note':'已完成导出及分片检查；vLLM加载健康检查由GPU对照任务完成，不代表精度无损。'}
    ready.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    (output / 'quantization_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(f'[quantization exported] {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
