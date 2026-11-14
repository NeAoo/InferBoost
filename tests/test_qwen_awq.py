"""Fail closed on missing routing branches; optional real Torch integration."""
import importlib.util
import unittest
from types import SimpleNamespace as NS

from pipeline.qwen_awq import inspect_fanouts, validate_resolved
from pipeline.awq_batching import reuse_evaluation_mode


class EvaluationModeContracts(unittest.TestCase):
    def test_nested_false_calls_and_failure_restore_original_method(self):
        class Model:
            def __init__(self): self.training, self.walks = False, 0
            def modules(self): return iter([self])
            def train(self, mode=True):
                self.walks += 1
                self.training = mode
                return self
        model = Model()
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with reuse_evaluation_mode(model):
                for _ in range(100): model.train(False)
                self.assertEqual(model.walks, 0)
                raise RuntimeError('probe')
        self.assertNotIn('train', vars(model))
        model.train(True)
        self.assertTrue(model.training)
        self.assertEqual(model.walks, 1)

    def test_training_model_is_rejected(self):
        model = NS(training=True, modules=lambda: iter([NS(training=True)]))
        with self.assertRaises(ValueError):
            with reuse_evaluation_mode(model): pass


class Shape:
    def __init__(self, *shape):
        self.shape, self.ndim = shape, len(shape)
    def numel(self):
        from math import prod
        return prod(self.shape)


def fixture():
    modules = {'model.layers.0.post_attention_layernorm': NS(weight=Shape(8)),
               'model.layers.0.mlp': NS(experts=object()),
               'model.layers.0.mlp.gate': NS(weight=Shape(2, 8)),
               'model.layers.0.mlp.shared_expert_gate': NS(weight=Shape(1, 8))}
    for expert in ['experts.0', 'experts.1', 'shared_expert']:
        for projection in ['gate_proj', 'up_proj']:
            modules['model.layers.0.mlp.' + expert + '.' + projection] = NS(weight=Shape(4, 8))
    model = NS(config=NS(model_type='qwen3_5_moe_text'), named_modules=lambda: list(modules.items()))
    return model, modules


class FanoutContracts(unittest.TestCase):
    def test_router_and_shared_gate_are_compensation_only(self):
        model, modules = fixture()
        group, = inspect_fanouts(model)
        self.assertEqual(group.experts, 2)
        self.assertEqual(len(group.consumers), 8)
        self.assertEqual(set(group.compensation_only),
                         {'model.layers.0.mlp.gate', 'model.layers.0.mlp.shared_expert_gate'})

    def test_missing_expert_or_router_is_rejected(self):
        for name in ['model.layers.0.mlp.gate', 'model.layers.0.mlp.experts.1.up_proj']:
            model, modules = fixture()
            del modules[name]
            with self.assertRaises(ValueError): inspect_fanouts(model)

    def test_unknown_structure_and_wrong_width_are_rejected(self):
        model, modules = fixture()
        model.config.model_type = 'unknown_qwen'
        with self.assertRaises(ValueError): inspect_fanouts(model)
        model, modules = fixture()
        modules['model.layers.0.mlp.gate'].weight = Shape(2, 9)
        with self.assertRaises(ValueError): inspect_fanouts(model)

    def test_resolved_mapping_cannot_omit_or_quantize_router(self):
        from dataclasses import asdict
        model, modules = fixture()
        group, = inspect_fanouts(model)
        resolved = NS(smooth_name=group.norm, balance_names=list(group.consumers),
                      balance_layers=[modules[name] for name in group.consumers])
        modifier = NS(_resolved_mappings=[resolved])
        manifest = {'fanouts': [asdict(group)]}
        validate_resolved(modifier, manifest)
        resolved.balance_names.remove(group.compensation_only[0])
        with self.assertRaisesRegex(ValueError, '遗漏'): validate_resolved(modifier, manifest)
        resolved.balance_names = list(group.consumers)
        modules[group.compensation_only[0]].quantization_scheme = object()
        with self.assertRaisesRegex(ValueError, '不能低比特'): validate_resolved(modifier, manifest)


@unittest.skipUnless(importlib.util.find_spec('llmcompressor'), 'requires quantization runtime')
class ActualQwenContracts(unittest.TestCase):
    def test_real_qwen_moe_module_families(self):
        from scripts.validate_qwen_awq import validate
        report = validate()
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['models']), 4)


if __name__ == '__main__': unittest.main()
