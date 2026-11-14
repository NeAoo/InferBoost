"""Small CPU structural and scaling checks against the installed Qwen classes.

Exercises the actual AWQ final scaling operation with fixed scales, without
quantization rounding, then proves the missing-router negative case is caught.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def validate():
    import torch
    import transformers
    from pydantic import PrivateAttr
    from compressed_tensors.quantization import apply_quantization_config
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq.dynamic_mappings import get_layer_mappings_from_model
    from llmcompressor.modeling.moe.linearize import linearize_moe
    from llmcompressor.modeling.offset_norm import norm_calibration_context
    from pipeline.qwen_awq import build_mappings, inspect_fanouts, validate_resolved
    from pipeline.quantization_catalog import ignore_patterns
    from pipeline.awq_batching import batch_tokenwise_cache
    from pipeline.qwen_awq import make_modifier
    from llmcompressor.pipelines.cache import IntermediatesCache
    from llmcompressor.core import active_session
    torch.set_num_threads(2)
    torch.manual_seed(71)
    results = []

    variants = [('Qwen2MoeConfig', 'Qwen2MoeForCausalLM'),
                ('Qwen3MoeConfig', 'Qwen3MoeForCausalLM'),
                ('Qwen3NextConfig', 'Qwen3NextForCausalLM'),
                ('Qwen3_5MoeTextConfig', 'Qwen3_5MoeForCausalLM')]
    for config_name, class_name in variants:
        options = dict(vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_experts=4, num_experts_per_tok=2, norm_topk_prob=True,
            moe_intermediate_size=32, shared_expert_intermediate_size=32,
            decoder_sparse_step=1, mlp_only_layers=[],
            linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=16, linear_value_head_dim=16,
            layer_types=['linear_attention', 'full_attention'], mtp_num_hidden_layers=0)
        config = getattr(transformers, config_name)(**options)
        model = getattr(transformers, class_name)(config).eval()
        linearize_moe(model)
        groups = inspect_fanouts(model)
        replay = model.get_submodule(groups[0].parent)
        cache = IntermediatesCache(None, 'cpu')
        for length in (3, 7, 4):
            cache.append({'hidden_states': torch.randn(1, length, groups[0].width)})
        with torch.no_grad():
            def outputs():
                values=[]
                for batch in cache:
                    value=replay(**batch)
                    values.append((value[0] if isinstance(value,tuple) else value).reshape(-1,groups[0].width))
                return torch.cat(values)
            unbatched = outputs()
            if not batch_tokenwise_cache(cache, token_limit=8):
                raise ValueError('Tokenwise batching was not applied')
            torch.testing.assert_close(outputs(), unbatched, atol=2e-6, rtol=2e-5)
            if [next(iter(batch.values())).value.shape[-2] for batch in cache.batch_intermediates] != [8, 6]:
                raise ValueError('Replay token boundaries are incorrect')
            modifier, _ = make_modifier(model)
            modifier._parent_args_cache[replay] = cache
            replay_outputs = modifier._run_samples(replay)
            replay_flat = torch.cat([value.reshape(-1, groups[0].width) for value in replay_outputs])
            torch.testing.assert_close(replay_flat, unbatched, atol=2e-6, rtol=2e-5)
            # A mask refers to the original example layout: never rebatch it.
            masked_cache = IntermediatesCache(None, 'cpu')
            for length in (3, 7, 4):
                masked_cache.append({'hidden_states': torch.randn(1, length, groups[0].width)})
            modifier._parent_args_cache[replay] = masked_cache
            state = active_session().state
            previous = state.loss_masks
            try:
                state.loss_masks = [torch.ones(1, length) for length in (3, 7, 4)]
                if len(modifier._run_samples(replay)) != 3 or getattr(masked_cache, '_inferboost_batched', False):
                    raise ValueError('Masked calibration was incorrectly rebatched')
            finally:
                state.loss_masks = previous
            # Also cover full expert subtree replay for mappings that select it.
            # Current single-target up->down mappings replay down_proj itself.
            expert = replay.experts[0]
            expert_cache = IntermediatesCache(None, 'cpu')
            expert_inputs = [torch.randn(length, groups[0].width) for length in (3, 7, 4)]
            for value in expert_inputs:
                expert_cache.append({'hidden_states': value})
            modifier._parent_args_cache[expert] = expert_cache
            expected = torch.cat([expert(value) for value in expert_inputs])
            actual = modifier._run_samples(expert)
            if len(actual) != 1 or not getattr(expert_cache, '_inferboost_batched', False):
                raise ValueError('Expert parent replay was not batched')
            torch.testing.assert_close(actual[0], expected, atol=2e-6, rtol=2e-5)
        with norm_calibration_context(model):
            mappings, manifest = build_mappings(model, get_layer_mappings_from_model(model))
            apply_quantization_config(model, QuantizationModifier(targets=['Linear'], scheme='W4A16',
                ignore=ignore_patterns({'adapter': {'moe': True}})).resolve_quantization_config())
            # Exercise the actual adapter, including its evaluation-mode scope.
            adapter, _ = make_modifier(model)
            class FixedScaleAWQ(type(adapter)):
                _scales = PrivateAttr()
                def _run_samples(self, parent, **kwargs):
                    return [torch.zeros(1)]
                def _compute_best_scale(self, mapping, outputs, originals):
                    return self._scales
            modifier = FixedScaleAWQ(mappings=mappings, offload_device='cpu')
            modifier._set_resolved_mappings(model)
            validate_resolved(modifier, manifest)
            # Quantization wrappers must not alter this scaling-only experiment.
            from compressed_tensors.quantization import disable_quantization
            model.apply(disable_quantization)
            max_error = 0.0
            for group in groups:
                norm, mlp = model.get_submodule(group.norm), model.get_submodule(group.parent)
                inputs = torch.randn(2, 7, group.width)
                with torch.no_grad():
                    before_input = norm(inputs)
                    before_output = mlp(before_input)
                    if isinstance(before_output, tuple): before_output = before_output[0]
                    before_logits = torch.nn.functional.linear(before_input, mlp.gate.weight)
                    modifier._scales = torch.linspace(.5, 2., group.width)
                    modifier._smooth_activation_stats = {group.norm: [torch.ones(group.width), 1]}
                    modifier._parent_args_cache = {mlp: []}
                    modifier._apply_smoothing(model)
                    if 'train' in vars(model) or any(module.training for module in model.modules()):
                        raise ValueError('Evaluation-mode scope did not restore the model')
                    after_input = norm(inputs)
                    after_logits = torch.nn.functional.linear(after_input, mlp.gate.weight)
                    after_output = mlp(after_input)
                    if isinstance(after_output, tuple): after_output = after_output[0]
                    torch.testing.assert_close(after_logits, before_logits, atol=2e-6, rtol=2e-5)
                    torch.testing.assert_close(after_output, before_output, atol=2e-6, rtol=2e-5)
                    if not torch.equal(before_logits.topk(2).indices, after_logits.topk(2).indices):
                        raise ValueError('Expert selection changed after compensated scaling')
                    # Deliberately undo only router compensation: this must differ.
                    broken_logits = torch.nn.functional.linear(after_input, mlp.gate.weight / modifier._scales)
                    if torch.allclose(before_logits, broken_logits, atol=2e-6, rtol=2e-5):
                        raise ValueError('Negative control did not detect missing router compensation')
                    max_error = max(max_error, (after_output-before_output).abs().max().item())
            results.append({'class': class_name, 'fanouts': len(groups), 'experts_per_layer': groups[0].experts,
                'max_output_abs_error': max_error, 'route_top2_unchanged': True, 'negative_control_detected': True,
                'batched_replay_equivalent': True, 'adapter_smoothing_and_eval_restore': True})
    return {'passed': True, 'scope': 'real classes with small FP32 weights; full model BF16 validation is separate',
            'models': results}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output', type=Path)
    args=parser.parse_args(); report=validate()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
