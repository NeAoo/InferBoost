"""Image dependency checks; this does not claim GPU inference validation."""
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
quantization='--quantization' in sys.argv
packages=['torch','vllm','transformers','compressed-tensors','pandas']
if quantization: packages += ['llmcompressor','datasets','accelerate','auto-round']
versions={name:metadata.version(name) for name in packages}
assert versions['vllm']=='0.33.0'
assert versions['torch']=='2.14.0'
assert versions['transformers']=='5.20.2'
assert versions['compressed-tensors']==('0.20.1' if quantization else '0.17.0')
if quantization:
 from llmcompressor.modifiers.gptq import GPTQModifier
 from llmcompressor.modifiers.transform.awq import AWQModifier
 from datasets import Dataset
 import transformers
 assert hasattr(transformers,'Qwen3_5ForConditionalGeneration')
else:
 import pandas
 import vllm
source=Path('/opt/venv/lib/python3.11/site-packages/vllm/v1/worker/gpu/spec_decode/eagle/utils.py')
assert 'ACCEL60_DRAFT_CONFIG_FIX' in source.read_text()
print(json.dumps({'scope':'image_imports_only','quantization':quantization,'versions':versions}))
