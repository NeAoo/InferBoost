"""LLM Compressor recipes plus read-only identities for historical exports."""
from dataclasses import asdict, dataclass
import re
from .models import inspect_model


@dataclass(frozen=True)
class QuantizationRecipe:
    id: str
    backend: str
    algorithm: str
    weights: str
    activations: str
    format: str
    minimum_sm: int
    status: str


RECIPES = {item.id: item for item in [
    QuantizationRecipe('original', 'none', 'none', 'source_precision', 'source_precision', 'safetensors', 80, 'reference'),
    QuantizationRecipe('awq4', 'gptqmodel', 'awq', 'INT4', 'BF16', 'awq', 80, 'historical_reuse_only'),
    QuantizationRecipe('gptq4', 'gptqmodel', 'gptq', 'INT4', 'BF16', 'gptq', 80, 'historical_reuse_only'),
    QuantizationRecipe('gptq8', 'gptqmodel', 'gptq', 'INT8', 'BF16', 'gptq', 80, 'historical_reuse_only'),
    QuantizationRecipe('compressor_awq4', 'llmcompressor', 'awq', 'INT4', 'BF16', 'compressed-tensors', 80, 'requires_model_validation'),
    QuantizationRecipe('compressor_gptq4', 'llmcompressor', 'gptq', 'INT4', 'BF16', 'compressed-tensors', 80, 'requires_model_validation'),
    QuantizationRecipe('compressor_gptq8', 'llmcompressor', 'gptq', 'INT8', 'BF16', 'compressed-tensors', 80, 'requires_model_validation'),
    QuantizationRecipe('compressor_awq_gptq4', 'llmcompressor', 'awq_gptq', 'INT4', 'BF16', 'compressed-tensors', 80, 'upstream_recipe_requires_gpu_validation'),
    QuantizationRecipe('compressor_w4afp8', 'llmcompressor', 'awq', 'INT4', 'FP8_dynamic_per_token', 'compressed-tensors', 90, 'experimental_requires_gpu_validation'),
    QuantizationRecipe('compressor_w8a8', 'llmcompressor', 'gptq', 'INT8', 'INT8_dynamic_per_token', 'compressed-tensors', 80, 'requires_model_validation'),
    QuantizationRecipe('compressor_fp8', 'llmcompressor', 'rtn', 'FP8', 'FP8_dynamic_per_token', 'compressed-tensors', 89, 'requires_model_validation'),
]}

FRESH_RECIPES = tuple(name for name, recipe in RECIPES.items() if recipe.backend == 'llmcompressor')

SCHEMES = {
    'compressor_awq4': 'W4A16', 'compressor_gptq4': 'W4A16',
    'compressor_gptq8': 'W8A16', 'compressor_awq_gptq4': 'W4A16',
    'compressor_w4afp8': 'W4AFP8', 'compressor_w8a8': 'W8A8',
    'compressor_fp8': 'FP8_DYNAMIC',
}

ADAPTERS = {
    'qwen3_5': {'family': 'qwen35_dense', 'hybrid': True, 'moe': False},
    'qwen3_5_text': {'family': 'qwen35_dense', 'hybrid': True, 'moe': False},
    'qwen3_5_moe': {'family': 'qwen35_moe', 'hybrid': True, 'moe': True},
    'qwen3_5_moe_text': {'family': 'qwen35_moe', 'hybrid': True, 'moe': True},
    'llama': {'family': 'llama', 'hybrid': False, 'moe': False},
    'qwen2': {'family': 'qwen2', 'hybrid': False, 'moe': False},
    'qwen3': {'family': 'qwen3', 'hybrid': False, 'moe': False},
    'qwen2_moe': {'family': 'qwen2_moe', 'hybrid': False, 'moe': True},
    'qwen3_moe': {'family': 'qwen3_moe', 'hybrid': False, 'moe': True},
    'qwen3_next': {'family': 'qwen3_next', 'hybrid': True, 'moe': True},
}


def inspect_source(path):
    identity, tensors = inspect_model(path)
    config = identity['config']
    if config.get('quantization_config'):
        raise ValueError('第一阶段要求完整精度模型；已有量化模型可进入第二阶段')
    text = config.get('text_config', config)
    model_type = config.get('model_type')
    adapter = ADAPTERS.get(model_type) or ADAPTERS.get(text.get('model_type'))
    if adapter is None:
        raise ValueError(f'尚无经过审查的模型适配器: {model_type}；不会套用通用AWQ映射')
    mtp = sorted(key for key in tensors if re.search(r'(^|\.)mtp[._]', key))
    protected = sorted(key[:-7] for key in tensors if key.endswith('.weight') and (
        '.linear_attn.in_proj_a.' in key or '.linear_attn.in_proj_b.' in key))
    return {'path': str(path), 'model_type': model_type, 'text_model_type': text.get('model_type'),
            'adapter': adapter, 'identity': identity, 'mtp_tensor_names': mtp,
            'mtp_policy': 'preserve_original_precision', 'protected_modules': protected,
            'tie_word_embeddings': bool(text.get('tie_word_embeddings', config.get('tie_word_embeddings', False))),
            'weight_bytes': sum(shard['bytes'] for shard in identity['shards'].values())}


def ignore_patterns(model):
    patterns = ['lm_head', 're:.*lm_head$', 're:.*embed_tokens.*', 're:.*visual.*',
                're:(?:^|.*\\.)mtp(?:[._].*)?']
    patterns += ['re:.*linear_attn.in_proj_[ab]$', 're:.*linear_attn.in_proj_ba$']
    if model['adapter']['moe']:
        patterns += ['re:.*mlp.gate$', 're:.*shared_expert_gate$']
    return patterns


def eligibility(recipe_id, model, capability=None):
    if recipe_id not in RECIPES:
        raise ValueError('未知量化配方: ' + recipe_id)
    recipe = RECIPES[recipe_id]
    reason = None
    if recipe.backend == 'gptqmodel' and model['adapter']['family'] not in ('qwen35_dense', 'qwen35_moe'):
        reason = '现有GPTQModel适配限定Qwen3.5；其他架构使用独立LLM Compressor后端验收'
    if model['model_type'] not in ('qwen3_5', 'qwen3_5_moe') and recipe.backend == 'gptqmodel':
        reason = '现有修复绑定Qwen3.5多模态封装类，文本变体须另行验证'
    if capability is not None and 10 * capability[0] + capability[1] < recipe.minimum_sm:
        reason = f'配方要求至少SM{recipe.minimum_sm}；当前GPU能力不足'
    if model['adapter']['moe'] and recipe.id == 'compressor_w4afp8':
        reason = '当前W4AFP8配方未验证混合MoE，不能直接套用dense配方'
    return {'recipe_info': asdict(recipe), 'eligible': reason is None, 'reason': reason,
            'fresh_quantization_allowed': recipe.backend in ('none', 'llmcompressor'),
            'mtp_policy': 'preserve_original_precision', 'ignore': ignore_patterns(model)}
