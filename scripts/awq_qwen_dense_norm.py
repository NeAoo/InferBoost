"""Pinned offset RMSNorm correction for dense Qwen3.5; no MoE fanout rewrite."""
from contextlib import contextmanager
import hashlib
import inspect
from importlib.metadata import version
from pathlib import Path

MODEL_SHA256='0e2cd8dc50885b2701d26b116c585eedcdc62a24080ec34345af55b963126ded'
SCALE_SHA256='6a06b767c194be14793125c48d307b55c07926f8c875154df05e06c422fe3dca'


def fix_metadata():
    return {'version':'awq-qwen35-dense-offset-fanout-v2','class':'Qwen3_5RMSNorm',
            'formula':'(1 + weight) / scale - 1','gdn_input_fanout':['in_proj_qkv','in_proj_z','in_proj_b','in_proj_a'],
            'patch_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def verify_environment():
    from gptqmodel.quantization.awq.quantize import scale
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    if version('gptqmodel')!='7.5.0' or version('transformers')!='5.14.0':
        raise ValueError('Dense norm patch requires pinned quantization environment')
    for obj,expected in [(scale,SCALE_SHA256),(Qwen3_5RMSNorm,MODEL_SHA256)]:
        if hashlib.sha256(Path(inspect.getsourcefile(obj)).read_bytes()).hexdigest()!=expected:
            raise ValueError('Dense norm dependency fingerprint mismatch')
    return scale,Qwen3_5RMSNorm


@contextmanager
def corrected_dense_norm_scaling():
    import torch
    from gptqmodel.looper import awq_processor
    module,norm_class=verify_environment()
    original=module.scale_ln_fcs
    original_apply=module.apply_scale
    original_processor_apply=awq_processor.apply_scale
    if original_processor_apply is not original_apply:raise RuntimeError('Unknown existing apply_scale patch')
    statistics={'corrected_calls':0,'compensated_gdn_groups':0,'compensated_gdn_branches':0}

    @torch.inference_mode()
    def apply_complete_fanout(layer,scales_list,input_feat_dict=None):
        expanded=[]
        for previous,names,scales,loss in scales_list:
            names=list(names)
            if previous=='input_layernorm' and 'linear_attn.in_proj_qkv' in names:
                for name in ['linear_attn.in_proj_z','linear_attn.in_proj_b','linear_attn.in_proj_a']:
                    weight=layer.get_submodule(name).weight
                    if weight.ndim!=2 or weight.shape[1]!=scales.numel():
                        raise ValueError('Unexpected GDN fanout shape')
                    if name not in names:
                        names.append(name);statistics['compensated_gdn_branches']+=1
                statistics['compensated_gdn_groups']+=1
            expanded.append((previous,tuple(names),scales,loss))
        return original_apply(layer,expanded,input_feat_dict)
    @torch.inference_mode()
    def corrected(norm,layers,scales):
        if type(norm) is not norm_class:return original(norm,layers,scales)
        layers=layers if isinstance(layers,list) else [layers]
        if scales.ndim!=1 or scales.shape!=norm.weight.shape or any(layer.weight.ndim!=2 or layer.weight.shape[1]!=scales.numel() for layer in layers):
            raise ValueError('Dense norm scaling shape mismatch')
        scales=scales.to(norm.weight.device)
        if not bool(torch.isfinite(scales).all()) or not bool((scales>0).all()):
            raise ValueError('Invalid scaling factors')
        norm.weight.copy_((norm.weight.float()+1).div(scales.float()).sub(1))
        for layer in layers:layer.weight.mul_(scales.view(1,-1))
        statistics['corrected_calls']+=1
    module.scale_ln_fcs=corrected
    module.apply_scale=apply_complete_fanout
    awq_processor.apply_scale=apply_complete_fanout
    try:yield statistics
    finally:
        module.scale_ln_fcs=original
        module.apply_scale=original_apply
        awq_processor.apply_scale=original_processor_apply
