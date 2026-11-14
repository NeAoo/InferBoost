"""Frozen official calibration inputs and explicit separation from evaluation."""
import json
import re
from pathlib import Path

from .common import digest, read_json

OFFICIAL_VERSION = 'pipeline-official-calibration-v1'
OFFICIAL_DATASET = 'mlabonne/open-perfectblend'
OFFICIAL_REVISION = 'af60f3c18201652a83a93f46fcfee1b646ba3df7'
OFFICIAL_RECIPE = 'https://github.com/vllm-project/llm-compressor/blob/18243db74dba8020bd9ba68b5f4af761d9c4d34b/examples/quantization_w4a16/qwen3_8_gptq_awq_example.py'


def official_defaults():
    return {'mode': 'official', 'dataset': OFFICIAL_DATASET, 'revision': OFFICIAL_REVISION,
            'count': 512, 'max_tokens': 4096, 'seed': 42,
            'raw_data': None}


def conversation_queries(rows):
    return {' '.join(message['content'].split()) for row in rows for message in row['messages']
            if message['role'] == 'user' and message['content'].strip()}


def assert_independent(calibration_cases, evaluation_rows):
    if conversation_queries(calibration_cases) & conversation_queries(evaluation_rows):
        raise ValueError('校准集和独立评分集存在相同用户输入')


def load_calibration(directory, tokenizer_sha256):
    directory = Path(directory)
    audit = read_json(directory / 'manifest.json')
    if audit.get('version') != OFFICIAL_VERSION:
        from scripts.quantize_checkpoint import load_calibration as historical_loader
        return historical_loader(directory, tokenizer_sha256)
    if (audit.get('tokenizer_sha256') != tokenizer_sha256 or audit.get('test_query_overlap') != 0
            or audit.get('source_dataset') != OFFICIAL_DATASET
            or not re.fullmatch(r'[0-9a-f]{40}', audit.get('source_revision', ''))):
        raise ValueError('官方校准来源、tokenizer或评分隔离记录不符')
    for name, key in [('cases.jsonl', 'cases_sha256'), ('tokens.jsonl', 'tokens_sha256')]:
        if digest(directory / name) != audit[key]:
            raise ValueError('官方校准文件SHA256不符')
    cases = [json.loads(line) for line in (directory / 'cases.jsonl').read_text().splitlines()]
    tokens = [json.loads(line) for line in (directory / 'tokens.jsonl').read_text().splitlines()]
    if len(cases) != len(tokens) or len(tokens) != audit['count']:
        raise ValueError('官方校准样本数量不符')
    seen, result = set(), []
    for case, row in zip(cases, tokens):
        ids = row.get('input_ids')
        if (case.get('id') != row.get('id') or row.get('id') in seen or not isinstance(ids, list)
                or not 0 < len(ids) <= audit['max_tokens'] or len(ids) != case.get('selected_tokens')
                or not all(type(token) is int and token >= 0 for token in ids)):
            raise ValueError('官方校准token、样本ID或长度不合法')
        seen.add(row['id'])
        result.append({'input_ids': ids, 'attention_mask': [1] * len(ids)})
    if sum(len(row['input_ids']) for row in result) != audit['total_tokens']:
        raise ValueError('官方校准token总数不符')
    return audit, result
