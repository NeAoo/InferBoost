"""Display algorithm and precision without inferring algorithms from file formats."""
from .quantization_catalog import RECIPES


PRECISION_NOTE = (
    '仅权重量化的默认激活精度为A16（本流程配方使用BF16）；'
    'A8/AFP8表示额外启用了激活量化，不是反量化时才限制精度。'
    'W/A描述被量化线性层的权重和激活输入，不代表累加精度、KV Cache精度或所有层的精度；'
    'MTP及受保护模块保持原精度。'
)

ALGORITHM_EXPLANATIONS = [
    ('GPTQ', '压缩权重时逐步补偿量化误差，让压缩后的计算结果尽量接近原模型。'),
    ('AWQ', '用校准数据观察重要通道，调整缩放后压缩权重，减少重要信息的损失。'),
    ('AWQ＋GPTQ', '先做AWQ缩放，再用GPTQ量化并补偿误差；是同一配方依次处理，不是一半层用一种算法。'),
    ('W / A', 'W是模型权重，A是推理中间产生的激活；W4A16表示权重4位、激活16位，W8A8表示二者均为8位。'),
]


def quantization_sort_key(row):
    """Keep the reference first, then algorithm families and requested precision order."""
    recipe = RECIPES.get(row['recipe'])
    if recipe is None:
        return (99, 99, row['recipe'], row.get('candidate', ''), row.get('dataset', ''))
    family = {'none': 0, 'gptq': 1, 'awq': 2, 'awq_gptq': 3, 'rtn': 4}[recipe.algorithm]
    precision = {('INT4', 'BF16'): 0, ('INT8', 'BF16'): 1,
                 ('INT8', 'INT8_dynamic_per_token'): 2,
                 ('INT4', 'FP8_dynamic_per_token'): 3,
                 ('FP8', 'FP8_dynamic_per_token'): 4}.get((recipe.weights, recipe.activations), 99)
    return (family, precision, row['recipe'], row.get('candidate', ''), row.get('dataset', ''))


def describe_quantization(recipe_id=None, *, precision=None):
    recipe = RECIPES.get(recipe_id)
    if recipe is None:
        return {'quantization_label': '算法未记录 · ' + (precision or '精度未记录'),
                'algorithm': '未记录', 'weight_precision': '未记录',
                'activation_precision': '未记录'}
    if recipe.id == 'original':
        return {'quantization_label': '未量化 · 原始精度', 'algorithm': '未量化',
                'weight_precision': '保持源模型精度', 'activation_precision': '保持源模型精度'}
    algorithm = {'awq': 'AWQ', 'gptq': 'GPTQ', 'awq_gptq': 'AWQ＋GPTQ', 'rtn': 'RTN'}[recipe.algorithm]
    weight_bits = '4' if recipe.weights == 'INT4' else '8'
    if recipe.activations == 'BF16':
        notation, activation = 'W' + weight_bits + 'A16', 'BF16（16位；仅权重量化）'
    elif recipe.activations.startswith('INT8'):
        notation, activation = 'W' + weight_bits + 'A8', 'INT8（动态逐token激活量化）'
    else:
        notation, activation = ('W4AFP8' if recipe.weights == 'INT4' else 'W8A8-FP8'), 'FP8（动态逐token激活量化）'
    return {'quantization_label': algorithm + '-' + notation,
            'algorithm': algorithm, 'weight_precision': recipe.weights,
            'activation_precision': activation}
