"""Actual vLLM loading audit for unquantized and quantized stage-one candidates."""
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

import torch
from vllm.v1.worker.gpu_worker import Worker
from pipeline.kernel_audit import loaded_quantization


class QuantizationAuditWorker(Worker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        expected = json.loads(Path(os.environ['EXPECTED_BF16_FILE']).read_text())
        protected, methods, attention = [], Counter(), Counter()
        for name, module in self.model_runner.model.named_modules():
            method, scheme, kernel = loaded_quantization(module)
            if method is not None:
                methods[type(method).__name__] += 1
                if scheme is not None:
                    methods['scheme:' + type(scheme).__name__] += 1
                if kernel is not None:
                    methods['kernel:' + type(kernel).__name__] += 1
            if hasattr(module, 'impl'):
                attention[type(module.impl).__name__] += 1
            if name.endswith('.in_proj_ba'):
                weight = module.weight.detach().cpu().contiguous()
                fingerprint = hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()
                key = name[name.find('layers.'):]
                if weight.dtype != torch.bfloat16 or expected.get(key) != fingerprint:
                    raise RuntimeError('保留投影加载不符: ' + name)
                protected.append({'name': name, 'dtype': str(weight.dtype), 'sha256': fingerprint})
        if len(protected) != len(expected):
            raise RuntimeError('保留投影加载数量不符')
        receipt = {'passed': True, 'scope': 'stage_one_target_only', 'mtp_enabled': False,
                   'protected': protected, 'methods': dict(methods),
                   'attention_implementations': dict(attention),
                   'versions': {name: version(name) for name in ['vllm', 'torch', 'transformers', 'compressed-tensors']}}
        Path(os.environ['LOADING_AUDIT_FILE']).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
        return result
