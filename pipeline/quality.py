"""Deterministic, reference-based scoring; unlabeled inputs stay unscored."""
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path


SCORERS = {'exact', 'number', 'json_exact', 'json_fields'}


def json_equal(actual, expected):
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(json_equal(actual[key], value) for key, value in expected.items())
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(json_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def load_dataset(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('评测数据必须是非空 JSON 数组或 JSONL')
    seen = set()
    for row in rows:
        uid = row.get('uid')
        if not isinstance(uid, str) or not uid or uid in seen:
            raise ValueError('评测数据需要唯一、非空字符串 uid')
        seen.add(uid)
        if not isinstance(row.get('group'), str) or not row['group']:
            raise ValueError('评测数据需要 group')
        messages = row.get('messages')
        if not isinstance(messages, list) or not messages or any(
            not isinstance(message, dict) or not isinstance(message.get('content'), str)
            or message.get('role') not in ('system', 'user', 'assistant', 'tool') for message in messages
        ):
            raise ValueError('messages 必须符合文本聊天消息格式')
        if 'expected' in row and row.get('scorer') not in SCORERS:
            raise ValueError('带标准答案的数据需要明确 scorer；不自动猜测评分方法')
        if 'expected' not in row and row.get('scorer'):
            raise ValueError('scorer 必须配套 expected')
    return rows


def score(row, response):
    if 'expected' not in row:
        return {'status': 'unscored', 'reason': '没有标准答案，不能计算准确率'}
    if response.get('status') != 'success':
        return {'status': 'failed_request', 'score': 0.0, 'reason': response.get('status')}
    answer = response.get('content', '').strip()
    expected, scorer = row['expected'], row['scorer']
    try:
        if scorer == 'exact':
            matched = answer == str(expected).strip()
        elif scorer == 'number':
            actual_number, expected_number = Decimal(answer), Decimal(str(expected))
            matched = actual_number.is_finite() and expected_number.is_finite() and actual_number == expected_number
        else:
            # A fenced block is accepted only when the dataset explicitly allows it.
            if row.get('allow_code_fence', False) and answer.startswith('```'):
                lines = answer.splitlines()
                if len(lines) >= 3 and lines[-1].strip() == '```':
                    answer = '\n'.join(lines[1:-1])
            actual = json.loads(answer)
            if scorer == 'json_exact':
                matched = json_equal(actual, expected)
            elif scorer == 'json_fields':
                matched = isinstance(actual, dict) and isinstance(expected, dict) and all(
                    key in actual and json_equal(actual[key], value) for key, value in expected.items())
            else:
                raise ValueError('未知评分器: ' + str(scorer))
    except (InvalidOperation, ValueError, TypeError):
        return {'status': 'scored', 'score': 0.0, 'reason': '回答不符合评分格式'}
    return {'status': 'scored', 'score': float(bool(matched)),
            'reason': '匹配标准答案' if matched else '不匹配标准答案'}


def summarize(rows, responses):
    indexed = {item['uid']: item for item in responses}
    if len(indexed) != len(responses):
        raise ValueError('同一评测条件下有重复 uid')
    details = []
    for row in rows:
        response = indexed.get(row['uid'], {'status': 'missing', 'content': ''})
        details.append({'uid': row['uid'], 'group': row['group'], 'expected': row.get('expected'),
                        'content': response.get('content', ''), **score(row, response)})
    scored = [item for item in details if item['status'] != 'unscored']
    complete = set(indexed) == {row['uid'] for row in rows} and all(item.get('status') == 'success' for item in responses)
    return {'requests_expected': len(rows), 'requests_received': len(responses),
            'complete': complete,
            'scored_count': len(scored), 'unscored_count': len(details) - len(scored),
            'score_pct': 100 * sum(item['score'] for item in scored) / len(scored) if scored and complete else None,
            'details': details}
