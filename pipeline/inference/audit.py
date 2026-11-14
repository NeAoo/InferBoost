"""Observe actual loaded kernels; keep MTP and protected projections verifiable."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import torch
from vllm.v1.worker.gpu_worker import Worker as BaseWorker
from pipeline.kernel_audit import loaded_quantization


def fingerprint(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class Worker(BaseWorker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        model = self.model_runner.model
        references = json.loads(os.environ['STAGE2_REFERENCES'])
        expected = json.loads(Path(references['protected']).read_text()) if references.get('protected') else {}
        protected, methods, kernels, attention, gdn = {}, Counter(), Counter(), Counter(), Counter()
        activation_schemas = []
        kernel_evidence_complete = True
        for name, module in model.named_modules():
            method, scheme, kernel = loaded_quantization(module)
            if method is not None:
                methods[type(method).__name__] += 1
                if scheme is not None:
                    methods['scheme:' + type(scheme).__name__] += 1
                    known_a8 = {'CompressedTensorsW8A8Int8': 'int', 'CompressedTensorsW8A8Fp8': 'float'}
                    if type(scheme).__name__ in known_a8:
                        descriptor = {'class': type(scheme).__name__, 'num_bits': '8',
                                      'type': known_a8[type(scheme).__name__], 'evidence': 'loaded_native_a8_scheme'}
                        if descriptor not in activation_schemas:
                            activation_schemas.append(descriptor)
                if kernel is not None:
                    kernels[type(kernel).__name__] += 1
                elif (not type(method).__name__.startswith('Unquantized')
                      and 'KVCache' not in type(method).__name__):
                    kernel_evidence_complete = False
                schema = getattr(method, 'input_schema', None)
                if schema is not None:
                    descriptor = {'class': type(schema).__name__}
                    for key in ['dtype', 'num_bits', 'type', 'dynamic']:
                        if hasattr(schema, key):
                            descriptor[key] = str(getattr(schema, key))
                    if descriptor not in activation_schemas:
                        activation_schemas.append(descriptor)
            if hasattr(module, 'impl'):
                attention[type(module.impl).__name__] += 1
            if hasattr(module, 'gdn_prefill_backend'):
                gdn[str(module.gdn_prefill_backend)] += 1
            if expected and name.endswith('.in_proj_ba'):
                key = name[name.find('layers.'):]
                weight = module.weight
                value = fingerprint(weight)
                if weight.dtype != torch.bfloat16 or expected.get(key) != value:
                    raise RuntimeError('保护投影加载不匹配: ' + name)
                protected[key] = value
        if set(protected) != set(expected):
            raise RuntimeError('保护投影缺失')
        output = {'passed': True, 'methods': dict(methods), 'kernels': dict(kernels),
                  'attention': dict(attention), 'gdn': dict(gdn), 'protected': protected,
                  'kernel_evidence_complete': kernel_evidence_complete,
                  'activation_schemas': activation_schemas,
                  'compilation': str(self.vllm_config.compilation_config),
                  'attention_config': str(self.vllm_config.attention_config)}
        if int(os.environ.get('STAGE2_MTP', '0')):
            draft = self.model_runner.get_draft_model() if hasattr(self.model_runner, 'get_draft_model') else self.model_runner.drafter.model
            target = getattr(model, 'language_model', model)
            reference = json.loads(Path(references['mtp_reference']).read_text())
            if draft is None or getattr(draft, 'quant_config', None) is not None:
                raise RuntimeError('BF16草稿模型不存在或继承了目标量化配置')
            fc = draft.model.fc.weight
            shared_embedding = draft.model.embed_tokens.weight.data_ptr() == target.model.embed_tokens.weight.data_ptr()
            shared_head = draft.lm_head.weight.data_ptr() == target.lm_head.weight.data_ptr()
            if (fc.dtype != torch.bfloat16 or fingerprint(fc) != reference['mtp_fc_sha256']
                    or not shared_embedding or not shared_head):
                raise RuntimeError('MTP权重或与主模型的共享关系不匹配')
            output['mtp'] = {'dtype': str(fc.dtype), 'fc_sha256': fingerprint(fc),
                             'shared_embedding': shared_embedding, 'shared_head': shared_head}
        Path(os.environ['STAGE2_AUDIT_FILE']).write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
        return result
