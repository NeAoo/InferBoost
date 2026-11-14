"""Create protected-weight audit references and a separate unquantized MTP loading view."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open


def read_weight_map(directory):
    index = directory / 'model.safetensors.index.json'
    if index.is_file():
        return json.loads(index.read_text())['weight_map']
    with safe_open(directory / 'model.safetensors', framework='pt', device='cpu') as stream:
        return {key: 'model.safetensors' for key in stream.keys()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--mtp-source',type=Path)
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    mapping=read_weight_map(args.model)
    def load(directory,index,key):
        with safe_open(directory/index[key],framework='pt',device='cpu') as stream:return stream.get_tensor(key)
    fingerprint=lambda tensor:hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    expected={}
    for name in mapping:
        if name.endswith('.linear_attn.in_proj_b.weight'):
            other=name.replace('.in_proj_b.weight','.in_proj_a.weight')
            b,a=load(args.model,mapping,name),load(args.model,mapping,other)
            if b.dtype!=torch.bfloat16 or a.dtype!=torch.bfloat16:raise ValueError('受保护权重不是BF16')
            key=name[name.index('layers.'):].replace('.in_proj_b.weight','.in_proj_ba')
            expected[key]=fingerprint(torch.cat([b,a],dim=0))
    if not expected and not args.audit_only:raise ValueError('未找到GDN保护权重')
    (args.output/'expected_bf16.json').write_text(json.dumps(expected,indent=2)+'\n')
    if args.audit_only:
        print(json.dumps({'protected_layers':len(expected),'scope':'target_model_only'}))
        return
    if args.mtp_source is None:raise ValueError('生成MTP视图需要--mtp-source')
    original=read_weight_map(args.mtp_source)
    selected={key:value for key,value in original.items() if key.startswith('mtp.')}
    if 'mtp.fc.weight' not in selected:raise ValueError('没有可用MTP权重')
    draft=args.output/'draft';draft.mkdir()
    config=json.loads((args.mtp_source/'config.json').read_text());config.pop('quantization_config',None)
    (draft/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    weight_map={}
    for key,name in selected.items():
        dest='source-'+name
        if not (draft/dest).exists():(draft/dest).symlink_to((args.mtp_source/name).resolve())
        weight_map[key]=dest
    shared_keys=['model.language_model.embed_tokens.weight']
    target_config=json.loads((args.model/'config.json').read_text())
    text_config=target_config.get('text_config',target_config)
    tied=bool(text_config.get('tie_word_embeddings',target_config.get('tie_word_embeddings',False)))
    if 'lm_head.weight' in mapping:
        shared_keys.append('lm_head.weight')
    elif not tied:
        raise ValueError('未共享词嵌入的模型缺少lm_head权重')
    if tied != bool(config.get('text_config',config).get('tie_word_embeddings',config.get('tie_word_embeddings',False))):
        raise ValueError('主模型和MTP来源的词嵌入共享配置不一致')
    for key in shared_keys:
        name=mapping[key];dest='target-'+name
        if not (draft/dest).exists():(draft/dest).symlink_to((args.model/name).resolve())
        weight_map[key]=dest
    (draft/'model.safetensors.index.json').write_text(json.dumps({'weight_map':weight_map},indent=2)+'\n')
    (args.output/'mtp_reference.json').write_text(json.dumps({'mtp_fc_sha256':fingerprint(load(args.mtp_source,original,'mtp.fc.weight'))},indent=2)+'\n')
    print(json.dumps({'protected_layers':len(expected),'mtp_tensors':len(selected),'tied_embeddings':tied,'draft':str(draft)}))


if __name__=='__main__':main()
