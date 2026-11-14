"""Restore a merged BF16 checkpoint in a new directory using the original base and LoRA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def sha256(path):
    hasher=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):hasher.update(block)
    return hasher.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ['root','base','adapter','output']:
        parser.add_argument('--'+flag,type=Path,required=True)
    args=parser.parse_args()
    root,base,adapter,output=[getattr(args,name).resolve() for name in ['root','base','adapter','output']]
    if root not in output.parents or output==base or output==adapter:
        raise ValueError('合并产物必须写入独立工作目录')
    if output.exists():raise FileExistsError('拒绝覆盖已有目录: '+str(output))
    if json.loads((base/'config.json').read_text()).get('quantization_config'):
        raise ValueError('合并输入必须是完整精度基座')
    identity={str(path):sha256(path) for path in [base/'config.json',base/'model.safetensors.index.json',
                                                adapter/'adapter_config.json',adapter/'adapter_model.safetensors']}
    output.mkdir(parents=True)
    receipt={'state':'merging','base':str(base),'adapter':str(adapter),'source_metadata_sha256':identity,
             'device':'cpu','exact_old_export_identity':'not_claimed','started_at':time.time()}
    status=output/'merge_status.json';status.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from peft import PeftModel
    from importlib.metadata import version
    torch.set_num_threads(min(6,os.cpu_count() or 1))
    print('MERGE_LOAD_BASE',str(base),flush=True)
    model=AutoModelForImageTextToText.from_pretrained(str(base),dtype=torch.bfloat16,
        device_map='cpu',attn_implementation='eager',local_files_only=True,low_cpu_mem_usage=True)
    print('MERGE_LOAD_LORA',str(adapter),flush=True)
    model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False,local_files_only=True)
    print('MERGE_APPLY',flush=True)
    model=model.merge_and_unload(safe_merge=True)
    print('MERGE_SAVE',str(output),flush=True)
    model.save_pretrained(str(output),safe_serialization=True,max_shard_size='5GB')
    AutoTokenizer.from_pretrained(str(base),local_files_only=True).save_pretrained(str(output))
    for name in ['preprocessor_config.json','processor_config.json','chat_template.jinja']:
        for candidate in [adapter/name,base/name]:
            if candidate.is_file():shutil.copy2(candidate,output/name);break
    if any(sha256(Path(path))!=expected for path,expected in identity.items()):
        raise RuntimeError('合并过程中来源发生变化')
    receipt.update(state='merged_not_quality_validated',finished_at=time.time(),
        versions={name:version(name) for name in ['torch','transformers','peft']},
        output_files={path.name:{'bytes':path.stat().st_size,'sha256':sha256(path)} for path in output.glob('*.safetensors')})
    status.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print('MERGE_OK',str(output),flush=True)


if __name__=='__main__':main()
