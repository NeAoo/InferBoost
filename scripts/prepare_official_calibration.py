"""Freeze the official PerfectBlend inputs on CPU; never load model weights."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.calibration import (OFFICIAL_DATASET, OFFICIAL_RECIPE, OFFICIAL_REVISION,
                                  OFFICIAL_VERSION, assert_independent)
from pipeline.calibration import load_calibration
from pipeline.common import digest, write_json
from pipeline.quality import load_dataset


def normalize_conversations(row):
    roles = {'human': 'user', 'gpt': 'assistant'}
    messages = [{'role': roles.get(message['from'], message['from']), 'content': message['value']}
                for message in row['conversations']]
    if not messages or any(message['role'] not in ('system', 'user', 'assistant')
                           or not isinstance(message['content'], str) for message in messages):
        raise ValueError('PerfectBlend对话格式异常，停止而不替换样本')
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test-data', type=Path, nargs='+', required=True)
    parser.add_argument('--revision', default=OFFICIAL_REVISION)
    parser.add_argument('--count', type=int, default=512)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--raw-data', type=Path, help='已下载且有来源清单的官方JSONL样本，可离线处理')
    args = parser.parse_args()
    if args.count < 1 or args.max_tokens < 1:
        raise ValueError('校准数量与长度必须为正整数')
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        raise ValueError('校准数据必须固定为完整revision SHA')
    if args.output.exists():
        raise ValueError('官方校准输出已存在，不覆盖已冻结的数据')
    from datasets import Dataset
    from transformers import AutoTokenizer
    from llmcompressor.args import DatasetArguments
    from llmcompressor.transformers.data.perfectblend import PerfectBlendDataset
    from llmcompressor.transformers.data.base import _preserve_tokenizer_state

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    settings = DatasetArguments(dataset='perfectblend', max_seq_length=args.max_tokens,
                                streaming=True, pad_to_max_length=False,
                                raw_kwargs={'revision': args.revision})
    manager = PerfectBlendDataset(settings, split='train', processor=tokenizer)
    print(json.dumps({'event': 'loading_official_calibration', 'dataset': OFFICIAL_DATASET,
                      'revision': args.revision, 'count': args.count, 'max_tokens': args.max_tokens}), flush=True)
    # Taking the first N in a stream matches train[:N] without downloading all shards.
    if args.raw_data:
        receipt = json.loads(args.raw_data.with_suffix('.manifest.json').read_text())
        if (receipt.get('source_dataset') != OFFICIAL_DATASET or receipt.get('source_revision') != args.revision
                or receipt.get('records_sha256') != digest(args.raw_data)):
            raise ValueError('离线官方样本来源或SHA256不符')
        all_records = [json.loads(line) for line in args.raw_data.read_text().splitlines()]
        if len(all_records) != receipt['count']:
            raise ValueError('离线官方样本数量不符')
        raw = all_records[:args.count]
    else:
        raw = list(manager.load_dataset().take(args.count))
    if len(raw) != args.count:
        raise ValueError('官方数据不足，不能悄悄缩小校准集')
    records = Dataset.from_list([dict(row, source_row=index) for index, row in enumerate(raw)]).shuffle(seed=args.seed)
    cases, tokens = [], []
    with _preserve_tokenizer_state(tokenizer):
        for row in records:
            messages = normalize_conversations(row)
            prepared = manager.dataset_template(row)
            ids = manager.tokenize(prepared)['input_ids']
            uid = 'perfectblend-' + str(row['source_row'])
            cases.append({'id': uid, 'messages': messages, 'selected_tokens': len(ids),
                          'source_row': row['source_row']})
            tokens.append({'id': uid, 'input_ids': ids})
    evaluation = {str(path): load_dataset(path) for path in args.test_data}
    for rows in evaluation.values():
        assert_independent(cases, rows)
    args.output.mkdir(parents=True)
    for name, rows in [('cases.jsonl', cases), ('tokens.jsonl', tokens)]:
        (args.output / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    write_json(args.output / 'manifest.json', {
        'version': OFFICIAL_VERSION, 'source_dataset': OFFICIAL_DATASET,
        'source_revision': args.revision, 'source_split': 'train[:' + str(args.count) + ']',
        'upstream_example': OFFICIAL_RECIPE, 'upstream_preprocessor': 'PerfectBlendDataset',
        'count': args.count, 'max_tokens': args.max_tokens, 'seed': args.seed,
        'total_tokens': sum(len(row['input_ids']) for row in tokens),
        'selection': 'first N source records, HF Dataset.shuffle(seed), frozen before quantization',
        'cases_sha256': digest(args.output / 'cases.jsonl'), 'tokens_sha256': digest(args.output / 'tokens.jsonl'),
        'tokenizer_sha256': digest(args.model / 'tokenizer.json'), 'test_query_overlap': 0,
        'evaluation_sha256': {name: digest(name) for name in evaluation},
        'is_evaluation_data': False, 'weights_loaded': False,
        'raw_snapshot': {'path': str(args.raw_data), 'sha256': digest(args.raw_data)} if args.raw_data else None,
        'versions': {name: version(name) for name in ['llmcompressor', 'datasets', 'transformers']},
    })
    load_calibration(args.output, digest(args.model / 'tokenizer.json'))
    print(json.dumps({'event': 'official_calibration_frozen', 'directory': str(args.output),
                      'count': len(tokens), 'total_tokens': sum(len(row['input_ids']) for row in tokens)}), flush=True)


if __name__ == '__main__':
    main()
