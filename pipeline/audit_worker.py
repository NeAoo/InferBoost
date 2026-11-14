"""Audit loaded protected weights once, outside all measured request windows."""
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from vllm.v1.worker.gpu_worker import Worker


class AuditedWorker(Worker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        model = self.model_runner.model
        expected = json.loads(Path(os.environ['EXPECTED_BF16_FILE']).read_text())
        protected = []
        methods = Counter()
        experts = []
        attention = Counter()
        gdn = Counter()
        for name, module in model.named_modules():
            method = getattr(module, 'base_quant_method', getattr(module, 'quant_method', None))
            if hasattr(module, 'impl'):
                attention[type(module.impl).__name__] += 1
            if hasattr(module, 'gdn_prefill_backend'):
                gdn[str(module.gdn_prefill_backend)] += 1
            if method is not None:
                methods[type(method).__name__] += 1
            if name.endswith('.in_proj_ba'):
                weight = module.weight.detach().cpu().contiguous()
                fingerprint = hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()
                key = name[name.find('layers.'):]
                if weight.dtype != torch.bfloat16 or expected.get(key) != fingerprint:
                    raise RuntimeError('Protected BF16 weight differs: ' + name)
                protected.append({'name': name, 'dtype': str(weight.dtype), 'sha256': fingerprint})
            if name.endswith('.mlp.experts') or name.endswith('.mlp.experts.routed_experts'):
                experts.append({'name': name, 'class': type(module).__name__,
                                'method': type(method).__name__ if method else None,
                                'backend': str(getattr(method, 'wna16_moe_backend', '')),
                                'kernel': str(getattr(method, 'kernel', ''))[:250]})
        if len(protected) != len(expected) or not expected:
            raise RuntimeError(f'Expected {len(expected)} protected BF16 projections; found {len(protected)}')
        versions = {}
        for package in ['vllm', 'torch', 'triton', 'transformers', 'flashinfer-python']:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        output = {'passed': True, 'versions': versions, 'methods': dict(methods),
                  'protected': protected, 'experts': experts,
                  'attention_implementations': dict(attention), 'gdn_prefill_backends': dict(gdn),
                  'compilation': str(self.vllm_config.compilation_config),
                  'attention': str(self.vllm_config.attention_config)}
        if os.environ.get('MTP_AUDIT_REFERENCE'):
            reference=json.loads(Path(os.environ['MTP_AUDIT_REFERENCE']).read_text())
            if hasattr(self.model_runner,'get_draft_model'):
                draft=self.model_runner.get_draft_model()
            else:
                draft=self.model_runner.drafter.model
            if draft is None:
                raise RuntimeError('MTP was requested but no loaded draft model exists')
            if getattr(draft,'quant_config',None) is not None:
                raise RuntimeError('Official BF16 MTP draft inherited a quantization configuration')
            target=getattr(model,'language_model',model)
            fc=draft.model.fc.weight.detach().cpu().contiguous()
            fingerprint=hashlib.sha256(fc.view(torch.uint8).numpy().tobytes()).hexdigest()
            shared_embedding=draft.model.embed_tokens.weight.data_ptr()==target.model.embed_tokens.weight.data_ptr()
            shared_head=draft.lm_head.weight.data_ptr()==target.lm_head.weight.data_ptr()
            if (fc.dtype!=torch.bfloat16 or fingerprint!=reference['mtp_fc_sha256']
                    or not shared_embedding or not shared_head):
                raise RuntimeError('Official MTP head or target embedding/lm_head sharing differs')
            output['mtp']={'class':type(draft).__name__,'fc_dtype':str(fc.dtype),
                           'fc_sha256':fingerprint,'embeddings_shared':shared_embedding,
                           'lm_head_shared':shared_head,'quant_config':None,
                           'draft_config_fix':os.environ.get('ACCEL60_DRAFT_CONFIG_FIX')=='1',
                           'configuration':str(self.vllm_config.speculative_config)}
        Path(os.environ['LOADING_AUDIT_FILE']).write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
        return result
