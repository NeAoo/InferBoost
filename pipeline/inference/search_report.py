"""Show measured screening coverage separately from full-cohort final results."""

from pipeline.common import read_json, write_json
from .execution import complete, phase_results
from .decision_trace import build as build_decision_trace
from . import mtp_metrics


PHASE_LABELS = {'attention': '注意力后端', 'kernels': '矩阵内核', 'graphs': '编译与CUDA Graph',
                'budgets': '调度预算', 'mtp': 'MTP', 'full-screen': '完整网格',
                'baseline-fallback': '基准回退'}


def collect(run, config, candidates, plan, final_report=None):
    records, phases = [], []
    order = list(PHASE_LABELS)
    paths = sorted((run / 'phases').glob('*.json'),
                   key=lambda path: order.index(path.stem) if path.stem in order else len(order))
    for path in paths:
        phase = read_json(path)
        if not isinstance(phase, dict) or phase.get('kind') != 'screen':
            continue
        name = phase.get('name', path.stem)
        grouped = {}
        for result in phase_results(run, name):
            grouped.setdefault(result['profile_id'], []).append(result)
        completed = 0
        for trial in phase['candidates']:
            results = grouped.get(trial['id'], [])
            passed = (len(results) == phase['repeats']
                      and all(complete(result, phase['samples']) for result in results))
            completed += int(passed)
            audit_path = run / 'trials' / name / trial['id'] / 'audit.json'
            audit = read_json(audit_path) if audit_path.is_file() else {}
            # Each raw repetition remains visible; incomplete results are never ranked here.
            records.append({'phase': name, 'profile': trial['id'], 'label': trial['label'],
                            'settings': {key: trial[key] for key in
                                         ('attention', 'kernel', 'compile', 'graph', 'budget', 'mtp')},
                            'samples_per_cache': phase['samples'], 'expected_repeats': phase['repeats'],
                            'complete': passed, 'results': [{key: value for key, value in result.items()
                                                           if key != '_file'} for result in results],
                            'observed_kernels': audit.get('kernels'),
                            'observed_compilation': audit.get('compilation'),
                            'observed_attention': audit.get('attention'),
                            'observed_gdn': audit.get('gdn'),
                            'observed_mtp': audit.get('mtp'),
                            'mtp_acceptance': mtp_metrics.summarize(results, trial['mtp'],
                                run / 'trials' / name / trial['id'] / 'server.log', phase['repeats']),
                            'audit_passed': audit.get('passed')})
        phases.append({'phase': name, 'candidates': len(phase['candidates']),
                       'complete_candidates': completed, 'samples_per_cache': phase['samples'],
                       'repeats': phase['repeats']})
    allowed, search = plan.get('capabilities', {}).get('allowed', {}), config['search']
    axes = [allowed.get('attention'), allowed.get('kernels'), search.get('graphs'),
            search.get('budgets'), allowed.get('mtp_depths')]
    grid_size = None
    if all(axes):
        grid_size = len(axes[0]) * len(axes[1]) * (len(axes[2]) + 1) * len(axes[3]) * len(axes[4])
    defaults = {}
    for trial in candidates.values():
        command = trial.get('command', [])
        def value(flag):
            return command[command.index(flag) + 1] if flag in command else None
        if command:
            defaults[trial['id']] = {
                'prefix_cache': '--enable-prefix-caching' in command,
                'chunked_prefill': '--enable-chunked-prefill' in command,
                'kv_cache_dtype': value('--kv-cache-dtype'),
                'tensor_parallel_size': value('--tensor-parallel-size'),
                'max_model_len': value('--max-model-len'),
                'max_num_seqs': value('--max-num-seqs'),
                'gpu_memory_utilization': value('--gpu-memory-utilization'),
            }
    evidence = {'strategy': search.get('strategy', 'unrecorded'), 'phases': phases, 'records': records,
            'compatible_grid_size': grid_size, 'screened_candidates': len(records),
            'complete_screened_candidates': sum(row['complete'] for row in records),
            'skipped': plan.get('capabilities', {}).get('skipped', []),
            'baseline': plan.get('baseline'), 'recorded_common_settings_by_profile': defaults,
            'origin_to_optimized_comparison': 'not_measured_by_this_stage',
            'prefix_cache_ablation': 'not_part_of_this_search',
            'warning': '筛选小样本只用于选出复测候选，不能混入完整复测排名；分阶段筛选不保证全局最优。'}
    evidence['objective_cache'] = search['objective_cache']
    evidence['decision_steps'] = build_decision_trace(evidence, config)
    if final_report is not None:
        evidence['final_validation'] = {
            'objective_cache': search['objective_cache'], 'rows': final_report['rows'],
            'recommendation': final_report['recommendation'],
            'recommended_trial': candidates.get(final_report['recommendation']),
            'quality_gates': final_report['quality_gates'],
            'mtp_acceptance': final_report.get('mtp_acceptance', {}),
        }
    return evidence


def render(output, evidence, model_label):
    from .decision_view import render as render_decision_view
    write_json(output / 'search-details.json', evidence)
    (output / 'search-details.html').write_text(render_decision_view(evidence, model_label))
    from .text_report import search_markdown
    (output / 'search-details.md').write_text(search_markdown(evidence, model_label))
