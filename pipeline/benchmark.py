"""HTTP/SSE measurements with explicit stream completion and cache protocols."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
import urllib.error
import urllib.request
from .common import append_json, write_json
from .terminal import Progress


def endpoint(base, route, body=None, timeout=10):
    request = urllib.request.Request(base + route,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return response.read().decode()


def reset_cache(base):
    response = endpoint(base, '/reset_prefix_cache', {})
    if response.strip() == 'false':
        raise RuntimeError('服务器拒绝重置前缀缓存')


def read_sse(lines):
    data = []
    for raw in lines:
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if data:
                yield '\n'.join(data)
                data = []
        elif line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
    if data:
        yield '\n'.join(data)


def request_one(base, row, settings, warmup=False):
    maximum = 16 if warmup else settings['max_tokens']
    sampling = row.get('sampling', {})
    body = dict(sampling.get('extra_body', {}))
    body.update(model='pipeline-model', messages=row['messages'], temperature=0, seed=0,
                max_tokens=maximum, stream=True, stream_options={'include_usage': True})
    body.setdefault('chat_template_kwargs', {'enable_thinking': False})
    fixed = settings.get('output_mode') == 'fixed' or warmup
    if fixed:
        body.update(min_tokens=maximum, ignore_eos=True)
    result = {'uid': row['uid'], 'group': row['group'], 'requested_max_tokens': maximum,
              'output_mode': 'fixed' if fixed else 'natural', 'started_at': time.time()}
    request = urllib.request.Request(base + '/v1/chat/completions',
        data=json.dumps(body, ensure_ascii=False).encode(), headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.perf_counter()
    first = last = None
    content, reasoning, calls = [], [], []
    usage, finish, done = None, None, False
    try:
        with opener.open(request, timeout=settings.get('timeout_seconds', 300)) as response:
            if response.headers.get_content_type() != 'text/event-stream':
                raise RuntimeError('响应不是 SSE')
            for data in read_sse(response):
                if data == '[DONE]':
                    done = True
                    break
                if time.perf_counter() - started > settings.get('deadline_seconds', 600):
                    raise TimeoutError('请求超过总时间预算')
                event = json.loads(data)
                if event.get('error'):
                    raise RuntimeError(str(event['error'])[:500])
                if event.get('usage'):
                    usage = event['usage']
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    thought = delta.get('reasoning') or delta.get('reasoning_content')
                    if delta.get('content') or thought or delta.get('tool_calls'):
                        now = time.perf_counter()
                        first = now if first is None else first
                        last = now
                    if delta.get('content'):
                        content.append(delta['content'])
                    if thought:
                        reasoning.append(thought)
                    if delta.get('tool_calls'):
                        calls.extend(delta['tool_calls'])
                    finish = choice.get('finish_reason') or finish
        if not done or not usage or first is None or finish is None:
            raise RuntimeError('SSE 不完整：必须有正文/推理/工具增量、usage、finish_reason 和 [DONE]')
        if row.get('input_tokens') is not None and usage['prompt_tokens'] != row['input_tokens']:
            raise RuntimeError('tokenizer 与冻结输入 token 数不一致')
        if fixed and usage['completion_tokens'] != maximum:
            raise RuntimeError('固定输出长度未满足')
        output_tokens = usage['completion_tokens']
        decode_seconds = last - first
        result.update(status='fixed_complete' if fixed else 'truncated' if finish == 'length' else 'success',
            usage=usage, input_tokens=usage['prompt_tokens'], output_tokens=output_tokens,
            ttft_s=first-started, decode_s=decode_seconds,
            decode_tps=(output_tokens-1)/decode_seconds if output_tokens>1 and decode_seconds>0 else None,
            tpot_ms=1000*decode_seconds/(output_tokens-1) if output_tokens>1 and decode_seconds>0 else None,
            finish_reason=finish)
    except Exception as error:
        detail = error.read(1000).decode(errors='replace') if isinstance(error, urllib.error.HTTPError) else ''
        result.update(status='error', error=str(error), http_error=detail)
    result.update(e2e_s=time.perf_counter()-started, done=done, content=''.join(content),
                  reasoning=''.join(reasoning), tool_deltas=calls)
    return result


def run_suite(base, rows, settings, directory, profile_id):
    total = len(rows) * len(settings['concurrencies']) * len(settings.get('cache_modes', ['cold']))
    completed = 0
    with Progress(profile_id + ' · 评测', total, unit='请求') as display:
        for concurrency in settings['concurrencies']:
            for group in dict.fromkeys(row['group'] for row in rows):
                selected = [row for row in rows if row['group']==group]
                for offset in range(0, len(selected), concurrency):
                    batch = selected[offset:offset+concurrency]
                    reset_cache(base)
                    for cache_mode in settings.get('cache_modes', ['cold']):
                        before = endpoint(base, '/metrics')
                        started = time.perf_counter()
                        with ThreadPoolExecutor(max_workers=concurrency) as executor:
                            results = list(executor.map(lambda row: request_one(base,row,settings), batch))
                        wall_seconds = time.perf_counter()-started
                        after = endpoint(base, '/metrics')
                        batch_id = f'{concurrency}-{group}-{offset}-{cache_mode}'
                        for result in results:
                            result.update(profile=profile_id, concurrency=concurrency, cache=cache_mode,
                                          batch_id=batch_id)
                            append_json(directory/'requests.jsonl',result)
                        append_json(directory/'batches.jsonl',{'id':batch_id, 'concurrency':concurrency,
                            'group':group, 'cache':cache_mode, 'wall_s':wall_seconds,
                            'requests':len(batch), 'metrics_before':before, 'metrics_after':after})
                        progress = {'profile':profile_id, 'concurrency':concurrency, 'group':group,
                                    'completed_in_group':offset+len(batch), 'group_count':len(selected),
                                    'cache':cache_mode, 'statuses':[r['status'] for r in results]}
                        completed += len(results)
                        progress.update(completed=completed, total=total, updated_at=time.time())
                        write_json(directory/'progress.json',progress)
                        display.update(completed, detail=f'{group} · {cache_mode} · 并发{concurrency}')
                        if any(result['status']=='error' for result in results):
                            raise RuntimeError('请求失败；保留结果并停止该组合')


def percentile(values, fraction):
    values = sorted(value for value in values if value is not None and math.isfinite(value))
    if not values:
        return None
    rank = (len(values)-1)*fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    return values[lower] + (values[upper]-values[lower])*(rank-lower)
