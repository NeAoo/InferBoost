"""Cold/warm measurement hook called by vLLM's native sweep runner."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from pipeline.benchmark import endpoint, request_one, reset_cache
from pipeline.common import append_json, read_json, write_json
from pipeline.quality import score
from . import mtp_metrics


def representative_rows(rows, count):
    if count >= len(rows):
        return rows
    groups = {}
    for row in rows:
        groups.setdefault(row['group'], []).append(row)
    selected = []
    position = 0
    while len(selected) < count:
        for values in groups.values():
            if position < len(values):
                # Alternate low/high positions so small screening retains long examples.
                index = position // 2 if position % 2 == 0 else len(values) - 1 - position // 2
                selected.append(values[index])
                if len(selected) == count:
                    return selected
        position += 1
    return selected


def execution_signature(audit, candidate):
    return hashlib.sha256(json.dumps({
        'methods': audit['methods'], 'kernels': audit['kernels'], 'attention': audit['attention'],
        'gdn': audit['gdn'], 'attention_mode': candidate['attention'],
        'compile': candidate['compile'], 'graph': candidate['graph'],
        'budget': candidate['budget'], 'mtp': candidate['mtp'],
        'activation_schemas': audit['activation_schemas'],
    }, sort_keys=True).encode()).hexdigest()


def measured_stats(results):
    count = len(results)
    return {'requests': count, 'successes': sum(row['status'] == 'success' for row in results),
            'errors': sum(row['status'] == 'error' for row in results),
            'truncated': sum(row['status'] == 'truncated' for row in results),
            **{key: sum(values) / len(values) if values else None
               for key in ['ttft_s', 'e2e_s', 'decode_tps']
               for values in [[row[key] for row in results if row['status'] == 'success' and row.get(key) is not None]]},
            'quality_score': quality_score(results),
            'quality_scored_count': sum('score' in row.get('quality', {}) for row in results),
            'quality_expected_count': count}


def quality_score(results):
    values = [row['quality']['score'] for row in results if 'score' in row.get('quality', {})]
    return sum(values) / len(values) if values else None


def warmup(base, rows, settings):
    for group in dict.fromkeys(row['group'] for row in rows):
        selected = max([row for row in rows if row['group'] == group], key=lambda row: len(json.dumps(row['messages'])))
        reset_cache(base)
        result = request_one(base, selected, settings, warmup=True)
        if result['status'] != 'fixed_complete':
            raise RuntimeError('预热失败: ' + str(result.get('error', result['status'])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', nargs='?', default='bench', choices=['bench', 'reset'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--percentile-metrics')  # Native sweep's compatibility arguments.
    parser.add_argument('--save-result', action='store_true')
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--result-filename')
    parser.add_argument('--num-prompts', type=int)
    args = parser.parse_args()
    run = args.run_dir
    config = read_json(run / 'config.json')
    base = 'http://127.0.0.1:' + str(config['server']['port'])
    if args.action == 'reset':
        reset_cache(base)
        return
    if args.result_dir is None or args.result_filename is None:
        raise ValueError('缺少官方sweep结果路径')
    destination = args.result_dir / args.result_filename
    active = read_json(run / 'active.json')
    phase = read_json(run / 'phases' / (active['phase'] + '.json'))
    candidate = next(item for item in phase['candidates'] if item['id'] == active['profile_id'])
    def progress(stage, completed=0, total=None):
        write_json(run / 'progress.json', {'phase': active['phase'], 'profile': candidate['label'],
                   'profile_id': candidate['id'], 'round': args.result_filename, 'stage': stage,
                   'completed': completed, 'total': total, 'updated_at': time.time()})
    directory = run / 'trials' / active['phase'] / candidate['id']
    audit = read_json(directory / 'audit.json')
    if not audit.get('passed'):
        raise RuntimeError('加载审计未通过')
    model = read_json(run / 'model.json')
    if model['activation_bits'] == 8:
        evidence = json.dumps(audit['activation_schemas']).lower()
        if not any(value in evidence for value in ['float8', 'int8', '"num_bits": "8"']):
            raise RuntimeError('无法从实际加载审计确认A8，不能以A16回退结果冒充所选精度')
    if candidate['kernel'] != 'auto' and audit['kernels']:
        if not any(candidate['kernel'].lower() in name.lower() for name in audit['kernels']):
            raise RuntimeError('请求的矩阵后端没有实际生效: ' + candidate['kernel'])
    signature = execution_signature(audit, candidate)
    registry_file = run / 'execution_signatures.json'
    registry = read_json(registry_file) if registry_file.exists() else {}
    owner = registry.get(signature)
    if (phase['kind'] == 'screen' and audit.get('kernel_evidence_complete')
            and owner and owner['profile_id'] != candidate['id']):
        write_json(destination, {'stage2_status': 'duplicate', 'profile_id': candidate['id'],
                                 'duplicate_of': owner['profile_id'], 'execution_signature': signature})
        return
    rows = read_json(run / 'inputs.json')
    rows = representative_rows(rows, args.num_prompts or phase['samples'])
    settings = config['benchmark']
    progress('warmup')
    warmup(base, rows, settings)
    progress('performance', 0, len(rows))
    round_directory = directory / Path(args.result_filename).stem
    if round_directory.exists():
        round_directory.rename(round_directory.with_name(round_directory.name + '.partial-' + str(time.time_ns())))
    round_directory.mkdir()
    mtp_before = mtp_metrics.capture(base, round_directory / 'mtp-before.prom') if candidate['mtp'] else None
    by_cache = {'cold': [], 'warm': []}
    for index, row in enumerate(rows):
        reset_cache(base)
        for cache in ['cold', 'warm']:
            response = request_one(base, row, settings)
            response.update(cache=cache, concurrency=1, profile=candidate['id'], quality=score(row, response))
            append_json(round_directory / 'requests.jsonl', response)
            by_cache[cache].append(response)
        progress('performance', index + 1, len(rows))
        if any(by_cache[cache][-1]['status'] == 'error' for cache in by_cache):
            raise RuntimeError('接口请求失败，保留逐请求记录并排除该轮')
    mtp_acceptance = (mtp_metrics.difference(mtp_before, mtp_metrics.capture(base, round_directory / 'mtp-after.prom'))
                      if candidate['mtp'] else mtp_metrics.unavailable('MTP关闭', 'disabled'))
    write_json(round_directory / 'mtp-acceptance.json', mtp_acceptance)
    stats = {cache: measured_stats(values) for cache, values in by_cache.items()}
    quality_file = run / 'quality_inputs.json'
    if quality_file.exists():
        quality_rows = representative_rows(read_json(quality_file), phase['samples'])
        quality_results = {'cold': [], 'warm': []}
        progress('quality', 0, len(quality_rows))
        for index, row in enumerate(quality_rows):
            reset_cache(base)
            for cache in ['cold', 'warm']:
                response = request_one(base, row, settings)
                response.update(cache=cache, quality=score(row, response))
                quality_results[cache].append(response)
                append_json(round_directory / 'quality.jsonl', response)
            progress('quality', index + 1, len(quality_rows))
        for cache in stats:
            stats[cache]['quality_score'] = quality_score(quality_results[cache])
            stats[cache]['quality_scored_count'] = sum('score' in row['quality'] for row in quality_results[cache])
            stats[cache]['quality_expected_count'] = len(quality_rows)
            stats[cache]['quality_complete'] = all(row['status'] == 'success' for row in quality_results[cache])
            if not stats[cache]['quality_complete']:
                raise RuntimeError('质量复核请求失败或截断，排除本轮')
    if all(values['successes'] == len(rows) for values in stats.values()):
        registry[signature] = {'profile_id': candidate['id'], 'phase': active['phase']}
        write_json(registry_file, registry)
    write_json(destination, {'stage2_status': 'complete', 'profile_id': candidate['id'],
                             'samples': len(rows), 'cache': stats, 'execution_signature': signature,
                             'mtp_acceptance': mtp_acceptance,
                             'requests_file': str(round_directory / 'requests.jsonl')})


if __name__ == '__main__':
    main()
