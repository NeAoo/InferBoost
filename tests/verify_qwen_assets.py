"""Run with the pinned inference Python; checks single/sharded and tied/untied exports."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import save_file


root = Path(__file__).resolve().parents[1]
checks = []
with tempfile.TemporaryDirectory() as temporary:
    for sharded in (False, True):
        for tied in (False, True):
            case = Path(temporary) / f'{sharded}-{tied}'
            target, source = case / 'target', case / 'source'
            target.mkdir(parents=True)
            source.mkdir()
            settings = {'text_config': {'tie_word_embeddings': tied}}
            target_weights = {
                'model.language_model.layers.0.linear_attn.in_proj_b.weight': torch.ones(2, 4, dtype=torch.bfloat16),
                'model.language_model.layers.0.linear_attn.in_proj_a.weight': torch.zeros(2, 4, dtype=torch.bfloat16),
                'model.language_model.embed_tokens.weight': torch.ones(8, 4, dtype=torch.bfloat16),
            }
            if not tied:
                target_weights['lm_head.weight'] = torch.zeros(8, 4, dtype=torch.bfloat16)
            for directory, weights in ((target, target_weights), (source, {'mtp.fc.weight': torch.ones(4, 8, dtype=torch.bfloat16)})):
                (directory / 'config.json').write_text(json.dumps(settings))
                filename = 'weights-001.safetensors' if sharded else 'model.safetensors'
                save_file(weights, directory / filename)
                if sharded:
                    (directory / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {key: filename for key in weights}}))
            output = case / 'assets'
            result = subprocess.run([sys.executable, str(root / 'scripts/prepare_qwen_assets.py'),
                                     '--model', str(target), '--mtp-source', str(source),
                                     '--output', str(output)], capture_output=True, text=True, check=True)
            receipt = json.loads(result.stdout)
            if receipt['protected_layers'] != 1 or receipt['tied_embeddings'] != tied:
                raise AssertionError('Asset identity differs')
            index = json.loads((output / 'draft/model.safetensors.index.json').read_text())['weight_map']
            if ('lm_head.weight' in index) == tied:
                raise AssertionError('Unexpected output-head mapping')
            for filename in index.values():
                if not (output / 'draft' / filename).is_file():
                    raise AssertionError('Dangling model reference')
            checks.append({'sharded': sharded, 'tied': tied, 'passed': True})
print(json.dumps({'asset_cases': checks}))
