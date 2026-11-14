"""Token-weighted speculative acceptance, with explicit measurement boundaries."""
import math
from pathlib import Path
import re
import urllib.error

from pipeline.benchmark import endpoint
from pipeline.common import digest


COUNTERS = {
    'drafts': 'vllm:spec_decode_num_drafts_total',
    'drafted_tokens': 'vllm:spec_decode_num_draft_tokens_total',
    'accepted_tokens': 'vllm:spec_decode_num_accepted_tokens_total',
}
PERFORMANCE_SCOPE = 'cold＋warm性能请求合计；不含预热和独立质量复核；服务由本任务独占'
LOG_SCOPE = '日志观察值：混合预热、性能请求和质量复核；可能遗漏退出前未打印窗口，不能拆分cold/warm或轮次'
EXPLANATION = ('接受率＝主模型接受的草稿token数 ÷ 提出的草稿token数。先累加计数再相除，'
               '不平均每个窗口的百分比。接受率高不一定更快，仍按完整请求E2E选型。')


def unavailable(reason, status='unavailable'):
    return {'status': status, 'acceptance_rate': None, 'reason': reason}


def parse(text):
    series = {key: {} for key in COUNTERS}
    for key, name in COUNTERS.items():
        pattern = re.compile(r'^' + re.escape(name) + r'(\{.*\})?\s+(\S+)(?:\s+\S+)?$')
        for line in text.splitlines():
            match = pattern.fullmatch(line.strip())
            if not match:
                continue
            labels = match[1] or ''
            value = float(match[2])
            if not math.isfinite(value) or value < 0 or not value.is_integer() or labels in series[key]:
                raise ValueError('MTP计数非法或同标签重复: ' + name)
            series[key][labels] = int(value)
    if not all(series.values()):
        raise ValueError('服务未暴露完整MTP计数器')
    if any(set(values) != set(series['drafts']) for values in series.values()):
        raise ValueError('MTP计数器的模型/engine标签不一致')
    return series


def capture(base, path):
    """Store the raw snapshot outside request timers; unavailable metrics do not abort inference."""
    try:
        raw = endpoint(base, '/metrics', timeout=10)
    except (OSError, urllib.error.URLError) as error:
        return unavailable('无法读取/metrics: ' + str(error))
    path = Path(path)
    path.write_text(raw)
    try:
        return {'status': 'captured', 'series': parse(raw), 'file': str(path), 'sha256': digest(path)}
    except ValueError as error:
        return {**unavailable(str(error)), 'file': str(path), 'sha256': digest(path)}


def counts(accepted, drafted, drafts=None):
    values = [accepted, drafted] + ([] if drafts is None else [drafts])
    if any(type(value) is not int or value < 0 for value in values) or accepted > drafted:
        return unavailable('草稿计数非法', 'invalid')
    if drafts == 0 and drafted:
        return unavailable('草稿轮次为0但草稿token不为0', 'invalid')
    return {'status': 'measured' if drafted else 'no_drafts',
            'accepted_tokens': accepted, 'drafted_tokens': drafted, 'drafts': drafts,
            'acceptance_rate': accepted / drafted if drafted else None,
            'mean_acceptance_length_with_bonus': 1 + accepted / drafts if drafts else None}


def difference(before, after):
    if before.get('status') != 'captured' or after.get('status') != 'captured':
        return unavailable('性能区间快照缺失: ' + '; '.join(item.get('reason', '') for item in (before, after)))
    totals = {}
    for key in COUNTERS:
        first, last = before['series'][key], after['series'][key]
        if set(first) != set(last):
            return unavailable('性能区间内模型/engine标签变化', 'invalid')
        increments = [last[label] - first[label] for label in first]
        if any(value < 0 for value in increments):
            return unavailable('性能区间内计数器重置，拒绝计算', 'invalid')
        totals[key] = sum(increments)
    return {**counts(totals['accepted_tokens'], totals['drafted_tokens'], totals['drafts']),
            'source': 'prometheus_delta', 'scope': PERFORMANCE_SCOPE,
            'snapshots': {'before': {k: v for k, v in before.items() if k != 'series'},
                          'after': {k: v for k, v in after.items() if k != 'series'}}}


def log_observation(path):
    path = Path(path)
    if not path.is_file():
        return unavailable('旧实验没有结构化计数，且未找到服务日志')
    accepted = drafted = windows = 0
    pattern = re.compile(r'Accepted: (\d+) tokens, Drafted: (\d+) tokens')
    with path.open(errors='replace') as stream:
        for line in stream:
            if 'SpecDecoding metrics:' not in line:
                continue
            match = pattern.search(line)
            if not match:
                continue
            current_accepted, current_drafted = map(int, match.groups())
            if current_accepted > current_drafted:
                return unavailable('日志中存在非法草稿计数', 'invalid')
            accepted += current_accepted
            drafted += current_drafted
            windows += 1
    if not windows:
        return unavailable('日志未打印可恢复的MTP整数计数')
    return {**counts(accepted, drafted), 'status': 'log_observation' if drafted else 'no_drafts',
            'source': 'server_log', 'scope': LOG_SCOPE, 'windows': windows,
            'file': str(path), 'sha256': digest(path)}


def summarize(results, depth, log_path, expected_repeats):
    if not depth:
        return unavailable('MTP关闭', 'disabled')
    structured = [result.get('mtp_acceptance') for result in results]
    if not any(item is not None for item in structured):
        return log_observation(log_path)
    if len(results) != expected_repeats or any(not item or item.get('status') not in ('measured', 'no_drafts')
                                              or item.get('source') != 'prometheus_delta' for item in structured):
        return unavailable('结构化计数缺少完整轮次或有无效快照，不混用日志补齐', 'incomplete')
    total = {key: sum(item[key] for item in structured) for key in COUNTERS}
    return {**counts(total['accepted_tokens'], total['drafted_tokens'], total['drafts']),
            'source': 'prometheus_delta', 'scope': PERFORMANCE_SCOPE, 'rounds': len(results)}


def label(value):
    value = value or unavailable('未采集')
    rate = value.get('acceptance_rate')
    if rate is not None:
        suffix = '（日志观察值）' if value.get('source') == 'server_log' else '（性能合计）'
        return f'{rate:.2%}' + suffix
    return {'disabled': '不适用（MTP关闭）', 'incomplete': '未完整采集',
            'invalid': '计数无效', 'no_drafts': '无草稿token'}.get(value['status'], '未采集')
