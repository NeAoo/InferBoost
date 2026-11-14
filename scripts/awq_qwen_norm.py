"""进程内修复 GPTQModel 7.5.0 对 Qwen3.5 MoE offset RMSNorm 的 AWQ 缩放。

只修精确类 Qwen3_5MoeRMSNorm：有效系数是 1+weight。
Gated/Llama/Gemma 仍调用原实现；补齐post-norm的两个未量化路由分支。
不改搜索分组、统计、裁剪、校准输入或缓存规则。
上下文退出即恢复；共享 site-packages 永远只读。
"""
import hashlib
import inspect
import json
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

PATCH_VERSION = 'awq-qwen35-moe-scaling-v2'
SCALE_SHA256 = '6a06b767c194be14793125c48d307b55c07926f8c875154df05e06c422fe3dca'
QWEN_SHA256 = '688d9a8f2830d6729cd2945563f38b710100c086565b97c27c94c96bd9716b9f'
PROCESSOR_SHA256 = '51be5c05b800fa07cfd6506646575280762762e2273ab0016ae460ad9528b42c'
ROUTER_BRANCHES = ('mlp.gate', 'mlp.shared_expert_gate')


def fix_metadata():
    return {'version': PATCH_VERSION, 'patch_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scale_source_sha256': SCALE_SHA256, 'qwen_source_sha256': QWEN_SHA256,
            'target_class': 'Qwen3_5MoeRMSNorm', 'formula': '(1 + weight) / scale - 1',
            'unquantized_fanout_compensation': list(ROUTER_BRANCHES),
            'calibration_changed': False, 'search_and_clipping_changed': False,
            'shared_environment_modified': False}


def verify_environment():
    from gptqmodel.quantization.awq.quantize import scale
    from gptqmodel.looper import awq_processor
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm
    if version('gptqmodel') != '7.5.0' or version('transformers') != '5.14.0':
        raise ValueError('AWQ norm 修复只支持已核验的 GPTQModel 7.5.0 / Transformers 5.14.0')
    for obj, expected in [(scale, SCALE_SHA256), (Qwen3_5MoeRMSNorm, QWEN_SHA256),
                          (awq_processor, PROCESSOR_SHA256)]:
        source = Path(inspect.getsourcefile(obj))
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('AWQ norm 依赖源码指纹不符；禁止套用旧补丁')
    return scale, Qwen3_5MoeRMSNorm


@contextmanager
def corrected_qwen_norm_scaling(report_path=None):
    import torch
    from gptqmodel.looper import awq_processor
    scale_module, qwen_norm_class = verify_environment()
    original = scale_module.scale_ln_fcs
    original_apply = scale_module.apply_scale
    original_processor_apply = awq_processor.apply_scale
    if original_processor_apply is not original_apply:
        raise RuntimeError('AWQ apply_scale已被其他补丁替换，拒绝叠加未知行为')
    if getattr(original, '_qwen_offset_fix', False):
        raise RuntimeError('AWQ norm 修复不能重复安装')
    statistics = {'version': PATCH_VERSION, 'corrected_calls': 0, 'delegated_calls': 0,
                  'compensated_moe_groups': 0, 'compensated_router_branches': 0}

    @torch.inference_mode()
    def apply_with_complete_fanout(module, scales_list, input_feat_dict=None):
        expanded = []
        for prev_name, names, scales, loss in scales_list:
            names = list(names)
            if (prev_name == 'post_attention_layernorm'
                    and type(module.get_submodule(prev_name)) is qwen_norm_class):
                # 搜索结束后只补偿两个未量化分支，不加入w_mean、搜索或裁剪集合。
                for name in ROUTER_BRANCHES:
                    branch = module.get_submodule(name)
                    weight = getattr(branch, 'weight', None)
                    if weight is None or weight.ndim != 2 or weight.shape[1] != scales.numel():
                        raise ValueError('Qwen MoE路由结构不符，不能安全补偿AWQ缩放')
                    if name not in names:
                        names.append(name)
                        statistics['compensated_router_branches'] += 1
                statistics['compensated_moe_groups'] += 1
            expanded.append((prev_name, tuple(names), scales, loss))
        return original_apply(module, expanded, input_feat_dict)

    @torch.inference_mode()
    def corrected(norm, linear_layers, scales):
        if type(norm) is not qwen_norm_class:
            statistics['delegated_calls'] += 1
            return original(norm, linear_layers, scales)
        layers = linear_layers if isinstance(linear_layers, list) else [linear_layers]
        if (scales.ndim != 1 or scales.shape != norm.weight.shape or not layers
                or any(layer.weight.ndim != 2 or layer.weight.shape[1] != scales.numel()
                       for layer in layers)):
            raise ValueError('AWQ norm 缩放维度与后续线性层不匹配')
        scales = scales.to(norm.weight.device)
        if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
            raise ValueError('AWQ 缩放必须全部有限且大于0')
        if any(layer.weight.device != norm.weight.device for layer in layers):
            raise ValueError('AWQ norm 与后续线性层必须在同一设备')
        # 在FP32完成offset运算，仅最终写回原dtype，避免BF16中间舍入。
        updated = (norm.weight.float() + 1).div(scales.float()).sub(1)
        if not bool(torch.isfinite(updated.to(norm.weight.dtype)).all()):
            raise ValueError('AWQ norm 缩放产生非有限参数')
        norm.weight.copy_(updated)
        for layer in layers:
            layer.weight.mul_(scales.view(1, -1))
            if not bool(torch.isfinite(layer.weight).all()):
                raise ValueError('AWQ 后续线性层缩放产生非有限参数')
        statistics['corrected_calls'] += 1
        if report_path is not None:
            with Path(report_path).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({**statistics, 'event': 'qwen_offset_norm_scaled',
                    'width': scales.numel(), 'dtype': str(norm.weight.dtype),
                    'scale_min': scales.min().item(), 'scale_max': scales.max().item()}) + '\n')

    corrected._qwen_offset_fix = True
    scale_module.scale_ln_fcs = corrected
    scale_module.apply_scale = apply_with_complete_fanout
    awq_processor.apply_scale = apply_with_complete_fanout
    try:
        yield statistics
    finally:
        scale_module.scale_ln_fcs = original
        scale_module.apply_scale = original_apply
        awq_processor.apply_scale = original_processor_apply
