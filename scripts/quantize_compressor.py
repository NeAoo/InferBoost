"""LLM Compressor adapter: frozen tokens, explicit ignored modules, real export."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.common import digest, inside, write_json
from pipeline.quantization_catalog import RECIPES, SCHEMES, eligibility, ignore_patterns, inspect_source
from pipeline.calibration import load_calibration


def validate_main_model_only(model):
    """Do not let a backend's model loader enroll MTP in calibration or AWQ scaling."""
    mtp_modules = [name for name, _ in model.named_modules()
                   if re.search(r'(?:^|\.)mtp(?:[._]|$)', name)]
    if mtp_modules:
        raise ValueError('量化加载视图包含MTP模块，停止以保护原始权重: ' + ', '.join(mtp_modules[:5]))


def build_recipe(recipe_id, ignored, model=None, audit_path=None):
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    recipe = RECIPES[recipe_id]
    scheme = SCHEMES[recipe_id]
    options = {'targets': ['Linear'], 'scheme': scheme, 'ignore': ignored}
    def awq(duo_scaling=True):
        if model is not None:
            from pipeline.qwen_awq import QWEN_TYPES, make_modifier, model_type
            if model_type(model) in QWEN_TYPES:
                return make_modifier(model, duo_scaling=duo_scaling, audit_path=audit_path)[0]
        return AWQModifier(duo_scaling=duo_scaling)
    if recipe.algorithm == 'gptq':
        return [GPTQModifier(**options)]
    if recipe.algorithm == 'awq_gptq':
        return [awq('both'), GPTQModifier(**options)]
    if recipe.algorithm == 'rtn':
        return [QuantizationModifier(**options)]
    return [awq(), QuantizationModifier(**options)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    if job.get('mtp_policy') != 'preserve_original_precision':
        raise ValueError('本组件不量化MTP')
    progress_path = args.job.parent / 'backend_progress.json'
    def progress(stage, message):
        write_json(progress_path, {'stage': stage, 'message': message, 'updated_at': time.time()})
    progress('preparing', '检查模型与校准集')
    source = Path(job['source'])
    output = inside(job['root'], job['output'])
    if output.exists():
        raise ValueError('输出目录已存在，不覆盖未完成的量化任务')
    model_info = inspect_source(source)
    if RECIPES[job['recipe']].backend != 'llmcompressor':
        raise ValueError('LLM Compressor执行器不接受历史后端配方')
    audit, calibration = load_calibration(Path(job['calibration']), digest(source / 'tokenizer.json'))
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.utils import load_context
    import transformers

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('量化子进程必须只看到一张可用GPU')
    admission = eligibility(job['recipe'], model_info, torch.cuda.get_device_capability())
    if not admission['eligible']:
        raise ValueError(admission['reason'])
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model_class = getattr(transformers, model_info['identity']['config']['architectures'][0])
    output.mkdir(parents=True)
    started = time.monotonic()
    from scripts.awq_activation_cache import memory_monitor
    # LLM Compressor otherwise loads MTP by default, even when quantization ignores it.
    # Its save_pretrained hook copies unloaded MTP directly from the source checkpoint.
    with memory_monitor(output), load_context(model_class, load_mtp=False):
        progress('loading', '加载原模型；总进度尚不可用')
        model = model_class.from_pretrained(str(source), dtype=torch.bfloat16,
                                            local_files_only=True, trust_remote_code=False)
        validate_main_model_only(model)
        recipe = build_recipe(job['recipe'], ignore_patterns(model_info), model,
                              output / 'awq_adapter.json')
        tokenizer = transformers.AutoTokenizer.from_pretrained(source, local_files_only=True)
        calibration_options = dict(dataset=Dataset.from_list(calibration),
                num_calibration_samples=len(calibration), max_seq_length=max(len(row['input_ids']) for row in calibration),
                shuffle_calibration_samples=False, concatenate_data=False, batch_size=1,
                moe_calibrate_all_experts=model_info['adapter']['moe'],
                pipeline='sequential', sequential_offload_device='cpu', dataloader_num_workers=0,
                log_dir=str(output / 'logs'))
        if RECIPES[job['recipe']].algorithm == 'rtn':
            # Dynamic FP8 requires no activation calibration; do not run a needless forward pass.
            calibration_options = {'log_dir': str(output / 'logs')}
        progress('quantizing', '执行校准与量化；详细层级进度见 quantization.log')
        oneshot(model=model, tokenizer=tokenizer, recipe=recipe, **calibration_options)
        # Upstream chooses pack-quantized, int-quantized or float-quantized from the scheme.
        progress('exporting', '导出量化权重')
        model.save_pretrained(str(output), save_compressed=True)
        tokenizer.save_pretrained(output)
    from pipeline.models import inspect_model
    progress('verifying', '核验原模型与导出产物')
    current_source, _ = inspect_model(source, payload_hashes=True)
    if current_source != job['source_identity']:
        raise ValueError('原模型在量化期间改变，拒绝标记完成')
    write_json(output / 'quantization_ready.json', {
        'status': 'exported_not_serving_validated', 'source': str(source),
        'source_identity': job['source_identity'],
        'recipe': job['recipe'], 'mtp_policy': job['mtp_policy'],
        'scheme': SCHEMES[job['recipe']],
        'calibration_consumed': RECIPES[job['recipe']].algorithm != 'rtn',
        'mtp_loaded_for_calibration': False,
        'recipe_components': [type(modifier).__name__ for modifier in recipe],
        'moe_calibrate_all_experts': model_info['adapter']['moe'],
        'ignore': ignore_patterns(model_info), 'elapsed_seconds': time.monotonic() - started,
        'calibration': audit, 'versions': {name: version(name) for name in ['llmcompressor', 'compressed-tensors', 'torch', 'transformers']},
    })

    progress('completed', '量化导出完成，后续仍需评测')


if __name__ == '__main__':
    main()
