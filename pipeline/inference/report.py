"""A single cold/warm means table, generated only from complete final repetitions."""
import csv
import html
from pathlib import Path
import statistics

from pipeline.common import digest, read_json, write_json
from pipeline.report import combined_summary_table
from pipeline.quantization_labels import PRECISION_NOTE, describe_quantization
from pipeline.report_layout import PAGE_END, page_start
from .execution import complete, phase_results
from . import search_report
from . import mtp_metrics
from .defaults import defaults_html


NEAR_TIE_RELATIVE_GAP = .01


def generate(run, output_dir=None):
    run = Path(run)
    output = Path(output_dir) if output_dir is not None else run
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(run / 'config.json')
    candidates = read_json(run / 'candidates.json')
    final = read_json(run / 'finalists.json')
    model = read_json(run / 'model.json')
    hardware = read_json(run / 'hardware.json')
    inputs = read_json(run / 'inputs.json')
    plan = read_json(run / 'plan.json')
    recipe_id = config.get('handoff', {}).get('quantization', {}).get('recipe')
    description = describe_quantization(recipe_id, precision=model['precision'])
    rows, excluded, acceptance = [], [], {}
    grouped = {}
    for result in phase_results(run, 'final'):
        grouped.setdefault(result['profile_id'], []).append(result)
    for profile_id in final['profiles']:
        results = grouped.get(profile_id, [])
        acceptance[profile_id] = mtp_metrics.summarize(results, candidates[profile_id]['mtp'],
            run / 'trials/final' / profile_id / 'server.log', config['search']['repeats'])
        if len(results) != config['search']['repeats'] or not all(complete(item, len(inputs)) for item in results):
            excluded.append({'profile_id': profile_id, 'reason': '缺少全部完整复测轮次，或存在失败/截断'})
            continue
        for cache in ['cold', 'warm']:
            quality = [result['cache'][cache]['quality_score'] for result in results]
            def mean_metric(key):
                values = [result['cache'][cache][key] for result in results if result['cache'][cache][key] is not None]
                return statistics.mean(values) if values else None
            rows.append({'profile': profile_id, 'label': candidates[profile_id]['label'], 'cache': cache,
                         **description,
                         'complete': True, 'requests': len(inputs) * len(results),
                         'ttft_s_mean': mean_metric('ttft_s'),
                         'e2e_s_mean': mean_metric('e2e_s'),
                         'decode_tps_mean': mean_metric('decode_tps'),
                         'quality_scored_requests': sum(result['cache'][cache].get('quality_scored_count', 0) for result in results),
                         'quality_expected_requests': sum(result['cache'][cache].get('quality_expected_count', len(inputs)) for result in results),
                         'quality_score': statistics.mean(quality) if all(value is not None for value in quality) else None})
    best = {}
    quality_gates = {row['profile']: True for row in rows}
    quality_inputs = run / 'quality_inputs.json'
    if config.get('quality', {}).get('dataset'):
        # Missing frozen data remains a failed quality contract. A supplied
        # request-only workload is valid for performance tuning, but unscored.
        requires_quality = (not quality_inputs.is_file()
                            or any('expected' in item for item in read_json(quality_inputs)))
    else:
        requires_quality = any('expected' in item for item in inputs)
    comparison_ready = True
    for cache in ['cold', 'warm']:
        choices = [row for row in rows if row['cache'] == cache]
        baseline = next((row for row in choices if row['profile'] == final['baseline_id']), None)
        if requires_quality and (baseline is None or baseline['quality_score'] is None):
            comparison_ready = False
        if baseline and baseline['quality_score'] is not None:
            for row in choices:
                if row['quality_score'] is None or row['quality_score'] + 1e-12 < baseline['quality_score']:
                    quality_gates[row['profile']] = False
    for cache in ['cold', 'warm']:
        choices = [row for row in rows if row['cache'] == cache and quality_gates[row['profile']]
                   and comparison_ready]
        if choices:
            best[cache] = min(choices, key=lambda row: (row['e2e_s_mean'], row['profile']))['profile']
    report = {'stage': 2, 'rows': rows, 'best_by_cache': best, 'mtp_acceptance': acceptance,
              'experience': config.get('experience', {}),
              'recommendation': best.get(config['search']['objective_cache']),
              'objective': '并发1、目标缓存状态、完整复测的平均E2E最低',
              'quality_scored': bool(rows) and all(row['quality_scored_requests'] == row['quality_expected_requests'] for row in rows),
              'quality_gates': quality_gates,
              'quality_comparison_ready': comparison_ready,
              'report_generator_sha256': digest(__file__),
              'source_run': str(run), 'precision_note': PRECISION_NOTE,
              'excluded': excluded, 'native_sweep': True,
              'scope': '仅为当前模型、GPU、数据和本次候选范围内的性能选择'}
    failures = set()
    for path in (run / 'official').rglob('*failure.json'):
        failure = read_json(path)
        profile_id = failure.get('serve_overrides', {}).get('profile_id')
        if profile_id:
            failures.add(profile_id)
    duplicates = {result['profile_id'] for result in phase_results(run) if result.get('stage2_status') == 'duplicate'}
    report['failed_candidates'] = sorted(failures)
    report['duplicate_candidates'] = sorted(duplicates)
    search = search_report.collect(run, config, candidates, plan, final_report=report)
    report['search_coverage'] = {key: value for key, value in search.items()
                               if key not in ('records', 'recorded_common_settings_by_profile',
                                              'decision_steps', 'final_validation')}
    write_json(output / 'report.json', report)
    with (output / 'summary.csv').open('w', newline='') as stream:
        fields = ['quantization_label', 'weight_precision', 'activation_precision',
                  'label', 'cache', 'ttft_s_mean', 'e2e_s_mean', 'decode_tps_mean']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    escape = html.escape
    count = len(candidates)
    document = page_start('第二步：推理加速完整复测结果')
    document += defaults_html()
    if config.get('experience', {}).get('smoke_only'):
        document += '<p class="notice">快速体验：只验证链路；本次缩减样本与搜索范围，不构成正式部署推荐。</p>'
    model_label = description['quantization_label']
    document += f'<p class="meta">{escape(model_label)} · {escape(hardware["gpu"])}<br>权重：{escape(description["weight_precision"])}；激活输入：{escape(description["activation_precision"])}<br>并发1 · cold / warm · {len(inputs)}条数据 × {config["search"]["repeats"]}轮完整复测 · 以下均为平均值</p>'
    document += '<p class="lead"><a href="search-details.html">先看五步选型路线：为什么选这个注意力、矩阵内核、Graph、预算和MTP？</a></p>'
    document += '<p><a href="report.md">Markdown版（可在终端阅读）</a></p>'
    baseline_label = candidates.get(final['baseline_id'], {}).get('label', '基准配置缺失')
    document += '<p><strong>基准范围：</strong>此表比较同一已选模型的配置。基准为 ' + escape(baseline_label) + '；不能解释为“关闭全部优化的原始BF16模型”。</p>'
    document += f'<p class="meta">本次记录{count}个配置，其中{search["screened_candidates"]}个进入初筛，最终{len(rows)//2}组完成完整复测。<a href="search-details.html">查看完整筛选过程、默认设置和未测试项</a>。auto与显式内核实际相同时合并。</p>'
    if failures or duplicates:
        document += f'<p class="meta">启动/请求失败{len(failures)}组；实际执行重复、已合并{len(duplicates)}组。详细原因保存在原始记录中。</p>'
    display_rows = [dict(row, label=row['label'] + ('（质量复核未达标）' if not quality_gates[row['profile']] else ''))
                    for row in rows]
    document += '<div class="table-wrap">' + combined_summary_table({'cache_modes': ['cold', 'warm'], 'rows': display_rows}) + '</div>'
    from .decision_view import acceptance_table
    document += acceptance_table(acceptance, candidates)
    if not comparison_ready:
        document += '<p>质量对照基准缺少完整复测，暂不推荐；请补齐基准后重新生成报告。</p>'
    recommendations = {}
    for cache, profile_id in best.items():
        recommendations.setdefault(profile_id, []).append(cache)
    for profile_id, caches in recommendations.items():
        document += '<p><strong>' + ' / '.join(caches) + ' 本轮推荐：' + escape(candidates[profile_id]['label']) + '</strong>（平均E2E最低）</p>'
    objective_choices = sorted([row for row in rows if row['cache'] == config['search']['objective_cache']
                                and quality_gates[row['profile']] and comparison_ready],
                               key=lambda row: (row['e2e_s_mean'], row['profile']))
    if len(objective_choices) >= 2 and objective_choices[0]['e2e_s_mean'] > 0:
        gap = objective_choices[1]['e2e_s_mean'] - objective_choices[0]['e2e_s_mean']
        relative_gap = gap / objective_choices[0]['e2e_s_mean']
        if relative_gap < NEAR_TIE_RELATIVE_GAP:
            document += f'<p class="meta">目标缓存状态下，前两名平均E2E仅差{gap*1000:.2f}毫秒（{relative_gap*100:.2f}%）。差距较小，按本轮均值选择默认配置，不代表稳定优劣。</p>'
    if not recommendations:
        document += '<p>没有满足全部复测要求的推荐方案，请查看失败记录。</p>'
    skipped = plan['capabilities']['skipped']
    if skipped:
        document += '<p class="meta">' + escape('；'.join(item['option'] + '：' + item['reason'] for item in skipped)) + '。</p>'
    if not report['quality_scored']:
        document += '<p class="meta">未提供完整标准答案时不计算准确率；所有逐请求输出均已保存，可供业务质量复核。</p>'
    else:
        document += '<p class="meta">已另行完成带标准答案的规则复核；质量请求不混入表内性能均值。</p>'
    document += '<p><small>TTFT为首个输出等待时间，E2E为整段回复总耗时，TPS为生成速度。时间越低越好，TPS越高越好。cold与warm均已完成内核预热；warm表示重复前缀测试，并不保证每个短前缀都能命中缓存。本轮最优不代表所有模型、GPU和业务的最优。</small></p>'
    document += '<details><summary>精度与编译说明</summary><p>' + escape(PRECISION_NOTE) + '</p><p>编译使用vLLM mode=3；Eager同时关闭编译和Graph。初筛过程与实际内核见五步选型页。</p></details>' + PAGE_END
    (output / 'report.html').write_text(document)
    from .text_report import final_markdown
    (output / 'report.md').write_text(final_markdown(report, candidates))
    search_report.render(output, search, model_label)
    return report
