"""Independent quality plus equal-weight repeated performance measurements."""
import json
import statistics
from .common import read_json, write_json


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def measure(base, rows, settings, directory, profile_id):
    from . import benchmark, runtime
    quality_dir = directory / 'quality'
    quality_dir.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'measurement.json', {'stage': 'quality', 'round': 1, 'repeats': 1})
    quality_settings = dict(settings, concurrencies=[1], cache_modes=['cold'], output_mode='natural')
    benchmark.run_suite(base, rows, quality_settings, quality_dir, profile_id + ' · 独立评分')
    repeats = settings.get('performance_repeats', 3)
    for index in range(repeats):
        target = directory / 'performance' / f'round-{index + 1}'
        target.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'measurement.json', {'stage': 'performance', 'round': index + 1, 'repeats': repeats})
        runtime.warmup(base, rows, settings, target)
        benchmark.run_suite(base, rows, settings, target, f'{profile_id} · 性能{index + 1}/{repeats}')
    write_json(directory / 'measurement.json', {'stage': 'completed', 'round': repeats, 'repeats': repeats})


def performance_summary(directory, rows, settings):
    expected_ids = {row['uid'] for row in rows}
    repeats = settings.get('performance_repeats', 3)
    operating_points = [(c, cache) for c in settings.get('concurrencies', [1])
                        for cache in settings.get('cache_modes', ['cold', 'warm'])]
    complete, rounds = True, []
    for index in range(repeats):
        records = read_records(directory / 'performance' / f'round-{index + 1}' / 'requests.jsonl')
        primary = []
        for concurrency, cache in operating_points:
            cohort = [r for r in records if r['concurrency'] == concurrency and r['cache'] == cache]
            expected_status = 'fixed_complete' if settings.get('output_mode') == 'fixed' else 'success'
            valid = len(cohort) == len(rows) and {r['uid'] for r in cohort} == expected_ids and all(r['status'] == expected_status for r in cohort)
            complete &= valid
            if concurrency == 1 and cache == 'cold':
                primary = cohort if valid else []
        metrics = {}
        for key in ('ttft_s', 'e2e_s', 'decode_tps'):
            values = [r[key] for r in primary if r.get(key) is not None]
            metrics[key] = statistics.mean(values) if len(values) == len(rows) else None
        rounds.append({'round': index + 1, **metrics})
    result = {'complete': bool(complete), 'rounds': rounds, 'repeats': repeats,
              'unique_samples': len(rows), 'requests_per_cache_c1': len(rows) * repeats}
    for source, target in [('ttft_s', 'ttft_s'), ('e2e_s', 'e2e_s'), ('decode_tps', 'tps')]:
        values = [r[source] for r in rounds if r[source] is not None]
        result[target] = statistics.mean(values) if complete and len(values) == repeats else None
        result[target + '_std'] = statistics.stdev(values) if complete and len(values) == repeats and repeats > 1 else None
    return result
