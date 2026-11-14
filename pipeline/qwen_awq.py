"""Structural AWQ fan-out adapter for supported Qwen dense and MoE models.

Quantization remains owned by LLM Compressor. This module makes every consumer
of a smoothed MoE norm explicit, including unquantized routing branches.
"""
from dataclasses import asdict, dataclass
import re

ADAPTER_VERSION = 'qwen-awq-fanout-v2'
QWEN_TYPES = frozenset({'qwen2', 'qwen2_moe', 'qwen3', 'qwen3_moe',
    'qwen3_next', 'qwen3_5', 'qwen3_5_text', 'qwen3_5_moe', 'qwen3_5_moe_text'})


@dataclass(frozen=True)
class Fanout:
    norm: str
    parent: str
    consumers: tuple[str, ...]
    compensation_only: tuple[str, ...]
    experts: int
    width: int


def model_type(model):
    config = model.config
    text = getattr(config, 'text_config', config)
    return getattr(text, 'model_type', getattr(config, 'model_type', None))


def inspect_fanouts(model):
    """Validate loaded/linearized modules, never infer support from a folder name."""
    if model_type(model) not in QWEN_TYPES:
        raise ValueError('没有该架构的Qwen AWQ结构适配器: ' + str(model_type(model)))
    modules = dict(model.named_modules())
    groups = []
    for norm_name, norm in modules.items():
        if not norm_name.endswith('.post_attention_layernorm'):
            continue
        parent = norm_name.rsplit('.', 1)[0] + '.mlp'
        mlp = modules.get(parent)
        if mlp is None or not hasattr(mlp, 'experts'):
            continue
        router_name = parent + '.gate'
        router = modules.get(router_name)
        width = norm.weight.numel()
        if router is None or not hasattr(router, 'weight') or router.weight.ndim != 2:
            raise ValueError('MoE路由器必须是带二维weight的模块: ' + router_name)
        if router.weight.shape[1] != width:
            raise ValueError('MoE路由器与归一化维度不匹配: ' + parent)
        projections, expert_parts = [], {}
        for name, module in modules.items():
            if not name.startswith(parent + '.'):
                continue
            leaf = name.rsplit('.', 1)[-1]
            if leaf not in ('gate_proj', 'up_proj', 'gate_up_proj'):
                continue
            if not hasattr(module, 'weight') or module.weight.ndim != 2:
                raise ValueError('MoE专家未正确展开为二维线性层: ' + name)
            if module.weight.shape[1] != width:
                raise ValueError('MoE分支输入维度不匹配: ' + name)
            projections.append(name)
            if name.startswith(parent + '.experts.'):
                expert_parts.setdefault(name.rsplit('.', 1)[0], set()).add(leaf)
        expected_experts = router.weight.shape[0]
        if len(expert_parts) != expected_experts or any(
                parts not in ({'gate_proj', 'up_proj'}, {'gate_up_proj'})
                for parts in expert_parts.values()):
            raise ValueError(f'MoE专家映射不完整: {parent}, {len(expert_parts)}/{expected_experts}')
        compensation = [router_name]
        shared_gate = parent + '.shared_expert_gate'
        if shared_gate in modules:
            gate = modules[shared_gate]
            if gate.weight.ndim != 2 or gate.weight.shape[1] != width:
                raise ValueError('共享专家门控维度不匹配: ' + shared_gate)
            if not any(name.startswith(parent + '.shared_expert.') for name in projections):
                raise ValueError('共享专家门控存在但投影缺失: ' + parent)
            compensation.append(shared_gate)
        # An exact list avoids accidentally matching a neighbouring layer.
        groups.append(Fanout(norm_name, parent, tuple(sorted(projections) + compensation),
                             tuple(compensation), expected_experts, width))
    return groups


def build_mappings(model, upstream_mappings):
    """Use the upstream public mapping API; no installed library is modified."""
    from llmcompressor.modifiers.transform.awq.mappings import AWQMapping
    groups = inspect_fanouts(model)
    if not groups:
        if model_type(model) in {'qwen2_moe', 'qwen3_moe', 'qwen3_next',
                                'qwen3_5_moe', 'qwen3_5_moe_text'}:
            raise ValueError('MoE配置没有识别到完整的专家/路由结构')
        return upstream_mappings, {'version': ADAPTER_VERSION, 'model_type': model_type(model),
                                   'scope': 'upstream_dense', 'fanouts': []}
    covered = {group.norm for group in groups}
    mappings = []
    module_names = dict(model.named_modules())
    for mapping in upstream_mappings:
        matches = [name for name in module_names if (
            re.match(mapping.smooth_layer[3:], name) if mapping.smooth_layer.startswith('re:')
            else name == mapping.smooth_layer)]
        overlaps = covered.intersection(matches)
        if not overlaps:
            mappings.append(mapping)
        else:
            # Mixed dense/MoE models retain mappings for their dense layers.
            for name in matches:
                if name not in covered:
                    mappings.append(AWQMapping(name, list(mapping.balance_layers),
                                               mapping.activation_hook_target))
    for group in groups:
        # Upstream scans the full module tree once for every target. Group the
        # expert projections to avoid O(layers * experts * modules) matching.
        # The explicit manifest remains authoritative and is checked after
        # resolution, so a broad or missing match still fails closed.
        projections = ('re:^' + re.escape(group.parent)
                       + r'\.(?:experts\..+|shared_expert)\.(?:gate_proj|up_proj|gate_up_proj)$')
        mappings.append(AWQMapping(group.norm,
            [projections, *group.compensation_only], 'gate'))
    return mappings, {'version': ADAPTER_VERSION, 'model_type': model_type(model),
                      'scope': 'complete_moe_fanout', 'fanouts': [asdict(group) for group in groups]}


def validate_resolved(modifier, manifest):
    resolved = {mapping.smooth_name: mapping for mapping in modifier._resolved_mappings}
    for group in manifest['fanouts']:
        mapping = resolved.get(group['norm'])
        if mapping is None or set(mapping.balance_names) != set(group['consumers']):
            raise ValueError('AWQ实际映射遗漏MoE分支: ' + group['norm'])
        modules = dict(zip(mapping.balance_names, mapping.balance_layers))
        for name in group['compensation_only']:
            if getattr(modules[name], 'quantization_scheme', None) is not None:
                raise ValueError('路由分支只允许等价补偿，不能低比特量化: ' + name)


def make_modifier(model, *, duo_scaling=True, audit_path=None):
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from llmcompressor.modifiers.transform.awq.dynamic_mappings import get_layer_mappings_from_model
    from .awq_batching import REPLAY_TOKEN_LIMIT
    mappings, manifest = build_mappings(model, get_layer_mappings_from_model(model))
    tokenwise_parents = {model.get_submodule(group['parent']) for group in manifest['fanouts']}
    for group in manifest['fanouts']:
        for name in group['consumers']:
            if name.startswith((group['parent'] + '.experts.', group['parent'] + '.shared_expert.')):
                # Complete expert subtrees are also token-independent when an
                # upstream mapping selects them as replay parents.
                tokenwise_parents.add(model.get_submodule(name.rsplit('.', 1)[0]))
    manifest['replay_policy'] = {'token_limit': REPLAY_TOKEN_LIMIT, 'samples_unchanged': True,
                                'scope': 'Linear and verified Qwen MoE MLP; unmasked loss only',
                                'evaluation_mode': 'once_per_smoothing_phase'}

    class AuditedQwenAWQModifier(AWQModifier):
        def _apply_smoothing(self, current_model):
            from llmcompressor.utils.helpers import calibration_forward_context
            from .awq_batching import reuse_evaluation_mode
            # Upstream enters eval_context for each mapping. A large MoE has
            # thousands of mappings: establish eval once while keeping every
            # upstream per-mapping cache, kernel and offload context unchanged.
            with calibration_forward_context(current_model), reuse_evaluation_mode(current_model):
                return super()._apply_smoothing(current_model)

        def _run_samples(self, parent):
            import torch
            from llmcompressor.core import active_session
            from .awq_batching import batch_tokenwise_cache
            state = active_session().state
            if (not state.loss_masks and
                    (isinstance(parent, torch.nn.Linear) or parent in tokenwise_parents)):
                batch_tokenwise_cache(self._parent_args_cache[parent])
            return super()._run_samples(parent)

        def on_calibration_start(self, state, event, **kwargs):
            result = super().on_calibration_start(state, event, **kwargs)
            validate_resolved(self, manifest)
            if audit_path is not None:
                from .common import write_json
                write_json(audit_path, {**manifest, 'resolved_verified': True,
                    'compensation_policy': 'BF16 routing branches participate in scaling only'})
            return result

    return AuditedQwenAWQModifier(mappings=mappings, duo_scaling=duo_scaling,
                                 offload_device='cpu'), manifest
