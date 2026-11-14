"""Build a small, public, explicitly labelled calibration set for workflow validation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from transformers import AutoTokenizer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.quality import load_dataset


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--test-data',type=Path,required=True)
    parser.add_argument('--count',type=int,default=64)
    parser.add_argument('--max-tokens',type=int,default=512)
    parser.add_argument('--profile',choices=['summary','mixed'],default='summary')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    test_texts={json.dumps(row['messages'],ensure_ascii=False,sort_keys=True) for row in load_dataset(args.test_data)}
    cases=[];tokens=[]
    subjects=['城市公园开放安排','家庭节水实践','公共图书馆借阅流程','自行车日常维护','社区志愿活动','信息分类方法','新员工入门课程','科学实验观察记录']
    for i in range(args.count):
        subject=subjects[i%len(subjects)]
        body='\n'.join(f'第{j+1}项记录：围绕{subject}，参与者提出建议{(i+j)%17+1}，计划分步骤检查材料、安排时间、完成记录，并在结束后整理经验。' for j in range(24))
        prompt=f'阅读资料并概括主要行动和注意事项。练习编号CAL-{i:03d}，这是虚构的通用校准材料，不包含真实业务信息。\n'+body
        if args.profile=='mixed':
            quantity=701+13*i
            owner=['青禾','远舟','秋实','星河'][i%4]
            records='\n'.join(f'样例{j:03d}：名称物料{(i+j)%23}，库存{quantity+j}，状态'+('待核验' if j%3 else '已登记')+f'，数列[{j+31},{j-17},{i+j}]。' for j in range(32))
            instructions=[
                f'只输出JSON对象，提取下方主记录的code、owner、stock，不输出说明。主记录：code=CALX{i:03d}，owner={owner}，stock={quantity}。其他资料仅作背景。',
                f'求库存变化后的结果，只输出整数。最初{quantity}件，调出{51+i}件，补进{29+2*i}件。下方其余记录属于别的物料。',
                f'将[{quantity+51},{quantity-37},{quantity+8}]从小到大排序，只输出一维JSON数组；不要Markdown。其余资料仅作背景。',
                f'按规则输出标签：主记录状态含“待核验”则输出“待办”，否则输出“完成”。主记录CALX{i:03d}：状态待核验。其余记录不参与分类。',
            ]
            prompt=f'独立校准练习CAL-MIX-{i:03d}，以下全部为合成材料。\n'+instructions[i%4]+'\n'+records
        messages=[{'role':'user','content':prompt}]
        if json.dumps(messages,ensure_ascii=False,sort_keys=True) in test_texts:raise ValueError('校准与测试重合')
        rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer(rendered,add_special_tokens=False)['input_ids']
        ids=ids[:args.max_tokens]
        cases.append({'id':f'cal-{i:03d}','messages':messages,'selected_tokens':len(ids)})
        tokens.append({'id':f'cal-{i:03d}','input_ids':ids})
    for name,rows in [('cases.jsonl',cases),('tokens.jsonl',tokens)]:
        (args.output/name).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={'version':'pipeline-synthetic-calibration-v1','count':len(cases),'max_tokens':args.max_tokens,
              'total_tokens':sum(len(row['input_ids']) for row in tokens),
              'cases_sha256':sha(args.output/'cases.jsonl'),'tokens_sha256':sha(args.output/'tokens.jsonl'),
              'tokenizer_sha256':sha(args.model/'tokenizer.json'),'test_query_overlap':0,
              'test_dataset_sha256':sha(args.test_data),
              'purpose':'engineering_integration_not_production_quality','selection':'first max_tokens of each deterministic synthetic example'}
    manifest['profile']=args.profile
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':main()
