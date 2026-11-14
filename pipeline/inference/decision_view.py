"""A readable decision path, with controlled comparisons and optional raw evidence."""
import json

from pipeline.report_layout import PAGE_END, escape, page_start
from .decision_trace import STAGES, duplicate_target, option_label
from . import mtp_metrics
from .defaults import defaults_html


def number(value, places=4):
    return '—' if value is None else f'{value:.{places}f}'


def route(evidence):
    steps = evidence.get('decision_steps', [])
    if not steps:
        return ''
    document = '<nav aria-label="五步筛选路线"><ol class="route">'
    for index, step in enumerate(steps, 1):
        leader = step['leader']
        selected = leader['option'] if leader else '暂无有效结论'
        alternatives = list(dict.fromkeys(item['option'] for item in step['retained']
                                         if leader and item['option'] != selected))
        note = '另保留 ' + '、'.join(alternatives) if alternatives else '查看比较依据'
        final_trial = evidence.get('final_validation', {}).get('recommended_trial')
        if step['phase'] == 'mtp' and final_trial:
            note = '最终搭配预算 ' + str(final_trial['budget'])
        if step['phase'] == 'graphs' and leader:
            selected = selected.replace('编译＋', '')
        document += ('<li><a href="#step-' + escape(step['phase']) + '">' + str(index) + ' · '
                     + escape(step['title']) + '<strong>' + escape(selected) + '</strong><small>'
                     + escape(note) + '</small></a></li>')
    return document + '</ol></nav>'


def acceptance_table(acceptance, candidates):
    if not any(item.get('status') != 'disabled' for item in acceptance.values()):
        return ''
    document = '<h3>MTP草稿token接受率</h3><p>' + escape(mtp_metrics.EXPLANATION) + '</p>'
    document += '<div class="table-wrap"><table class="acceptance-table"><thead><tr><th>方案</th><th>接受率</th><th>接受 / 草稿token</th><th>统计范围</th></tr></thead><tbody>'
    for profile, item in acceptance.items():
        document += '<tr><td>' + escape(candidates.get(profile, {}).get('label', profile)) + '</td><td>' + escape(mtp_metrics.label(item)) + '</td><td>'
        document += str(item.get('accepted_tokens', '—')) + ' / ' + str(item.get('drafted_tokens', '—'))
        document += '</td><td>' + escape(item.get('scope', item.get('reason', '未采集'))) + '</td></tr>'
    return document + '</tbody></table></div>'


def _comparison(group, objective, show_acceptance=False):
    document = '<p class="fixed">保持不变：' + escape(group['fixed_label']) + '</p>'
    document += '<div class="table-wrap"><table><thead><tr><th>本步选项</th><th>cold 总耗时秒</th><th>warm 总耗时秒</th><th>warm 首字秒</th><th>warm TPS</th>'
    document += ('<th>草稿token接受率</th>' if show_acceptance else '') + '<th>本步结论</th></tr></thead><tbody>'
    for row in group['rows']:
        cold, warm = row['cold'] or {}, row['warm'] or {}
        winner = row['profile'] == group['best_id']
        if row.get('duplicate_of'):
            conclusion = '实际执行相同，已合并；参见 ' + (row.get('duplicate_label') or row['duplicate_of'])
        elif not row['complete']:
            conclusion = '未完成，不参与选择'
        elif not row['quality_eligible']:
            conclusion = '质量或数据条件未达标'
        elif winner:
            conclusion = '本组均值最低'
        elif row['retained']:
            conclusion = '保留为备选'
        else:
            conclusion = '本组耗时较高'
        document += '<tr' + (' class="winner"' if winner else '') + '><td>' + escape(row['option'])
        if row['settings']['kernel'] == 'auto' and row['option'].startswith('auto'):
            kernels = row.get('observed_kernels') or {}
            actual = next((label for name, label in [('CutlassInt8ScaledMMLinearKernel', 'CUTLASS INT8'),
                                                    ('HummingInt8ScaledMMLinearKernel', 'Humming INT8')]
                           if name in kernels), None)
            if actual:
                document += '<span class="state-note">本轮实际：' + actual + '</span>'
        if row['reused']:
            document += '<span class="state-note">复用前面已测成绩</span>'
        document += '</td>' + ''.join('<td>' + number(value, places) + '</td>' for value, places in
                                     ((cold.get('e2e_s'), 4), (warm.get('e2e_s'), 4),
                                      (warm.get('ttft_s'), 4), (warm.get('decode_tps'), 2)))
        if show_acceptance:
            document += '<td>' + escape(mtp_metrics.label(row.get('mtp_acceptance'))) + '</td>'
        document += '<td>' + escape(conclusion) + '</td></tr>'
    document += '</tbody></table></div>'
    if group['relative_gap'] is not None and group['relative_gap'] < .01:
        document += '<p class="caption">前两名的 ' + escape(objective) + ' 平均总耗时相差 ' + f'{group["relative_gap"]*100:.2f}%' + '，差距较小；按本轮均值排序，不代表稳定优劣。</p>'
    return document


def render(evidence, model_label):
    steps = evidence.get('decision_steps', [])
    final = evidence.get('final_validation', {})
    selected = final.get('recommended_trial')
    document = page_start('第二步：看懂加速方案怎么选')
    document += defaults_html()
    document += '<p class="lead">先读路线，再看每一步的对照表，最后检查完整复测结果。</p>'
    document += '<p class="meta">模型：' + escape(model_label) + '。选择目标：请求完整，有标准答案时核对质量，再优先降低 ' + escape(evidence.get('objective_cache', '未记录')) + ' 平均总耗时（E2E）。</p>'
    if selected:
        document += '<p class="config-line">最终推荐：' + escape(selected['label']) + '</p>'
    document += '<div class="inline-links"><a href="report.html">查看完整复测结果</a><a href="search-details.md">Markdown版</a><a href="#defaults">默认设置与作用</a><a href="#raw-records">全部原始记录</a></div>'
    document += route(evidence)
    if steps:
        document += '<p class="caption">上图表示初筛过程。每一步保留领先的少数组合继续测试，因此早期领先选项可能在后续交叉测试中改变；点击任一步查看依据。</p>'
    else:
        document += '<p>本批次没有可重建的分阶段筛选记录；不将固定配置验证或完整网格改写成五步选型。</p>'
    document += '<div class="notice"><p>cold：清除前缀缓存后请求；warm：立即重复同一请求。二者都已经预热。每一步只改变表内选项，其他条件写在表上方；初筛成绩与最终完整复测分开。</p></div>'
    for index, step in enumerate(steps, 1):
        name = step['phase']
        document += '<section class="stage" id="step-' + escape(name) + '"><h2>' + str(index) + '．' + escape(step['title']) + '</h2>'
        document += '<p>' + escape(step['purpose']) + '</p>'
        if step.get('options_help'):
            document += '<p class="caption">' + escape(step['options_help']) + '</p>'
        for skipped in step.get('skipped', []):
            document += '<p class="notice"><strong>' + escape(skipped['option']) + '：本轮跳过。</strong> '
            if name == 'kernels':
                document += '所选模型为 ' + escape(model_label) + '。'
            document += escape(skipped['reason']) + '；未执行，不列入耗时排名。</p>'
        document += '<p class="caption">初筛：每种缓存状态 ' + str(step['samples_per_cache']) + ' 条 × ' + str(step['repeats']) + ' 轮；按 ' + escape(step['objective_cache']) + ' 总耗时选择。以下主表展示本步领先组合的同条件比较。</p>'
        if step['groups']:
            document += _comparison(step['groups'][0], step['objective_cache'], name == 'mtp')
        if name == 'mtp':
            document += '<p class="caption">' + escape(mtp_metrics.EXPLANATION) + '</p>'
            document += '<p class="caption">标记“性能合计”的新实验：' + escape(mtp_metrics.PERFORMANCE_SCOPE) + '。</p>'
            document += '<p class="caption">历史实验：' + escape(mtp_metrics.LOG_SCOPE) + '。</p>'
        leader = step['leader']
        if leader:
            criterion = '完整且符合质量条件的结果' if step.get('quality_reference') is not None else '完整有效的性能结果'
            document += '<p class="decision"><strong>本步领先：' + escape(leader['option']) + '。</strong> 在以上固定条件下，' + criterion + '中，目标缓存的平均总耗时最低。</p>'
            if step.get('quality_reference') is None:
                document += '<p class="caption">本步没有可用的基准质量分数，仅比较性能，不代表准确率已验证。</p>'
        else:
            document += '<p>没有完整且符合条件的结果，不能作出本步选择。</p>'
        if selected and leader and option_label(name, selected) != leader['option']:
            document += '<p class="choice">后续测试调整了这一项：本步是 ' + escape(leader['option']) + ' 领先，最终配置采用 ' + escape(option_label(name, selected)) + '；最终结论来自后面的组合测试与完整复测。</p>'
        if step['retained']:
            document += '<details><summary>继续验证的 ' + str(len(step['retained'])) + ' 组配置'
            if len(step['groups']) > 1:
                document += '，以及其他固定条件下的比较'
            document += '</summary><ul>' + ''.join('<li>' + escape(item['label']) + '</li>' for item in step['retained']) + '</ul>'
            for group in step['groups'][1:]:
                document += _comparison(group, step['objective_cache'], name == 'mtp')
            document += '</details>'
        document += '</section>'
    document += '<section class="stage"><h2>最后：用完整数据确认推荐</h2>'
    if final.get('rows'):
        document += '<p>前面的表用于初筛；下面单独展示完整复测，结果不与初筛混算。最终按 ' + escape(final['objective_cache']) + ' 平均总耗时推荐，质量不达标或缺少轮次的候选不参与推荐。</p><div class="table-wrap"><table><thead><tr><th>最终方案</th><th>TTFT秒</th><th>E2E秒</th><th>TPS估计</th><th>性能请求数</th><th>结论</th></tr></thead><tbody>'
        for row in final['rows']:
            if row['cache'] != final['objective_cache']:
                continue
            recommended = row['profile'] == final.get('recommendation')
            conclusion = '本轮推荐' if recommended else '质量复核未通过' if not final.get('quality_gates', {}).get(row['profile'], True) else '完整复测对照'
            document += '<tr' + (' class="winner"' if recommended else '') + '><td>' + escape(row['label']) + '</td>'
            document += ''.join('<td>' + number(row.get(key), places) + '</td>' for key, places in [('ttft_s_mean', 4), ('e2e_s_mean', 4), ('decode_tps_mean', 2)])
            document += '<td>' + str(row['requests']) + '</td><td>' + conclusion + '</td></tr>'
        document += '</tbody></table></div><p class="caption">每种缓存状态的请求数分别统计；完整cold/warm对照见结果页。总耗时更低不保证首字更快，最终仍由用户按业务目标选择。</p>'
        document += acceptance_table(final.get('mtp_acceptance', {}),
                                     {row['profile']: row for row in final['rows']})
    else:
        document += '<p>尚无完整复测结果，暂不提供最终推荐。</p>'
    document += '</section>'
    document += '<h2>本轮结论的范围</h2><p>完成初筛 ' + str(evidence['complete_screened_candidates']) + ' / ' + str(evidence['screened_candidates']) + ' 组。'
    if evidence.get('compatible_grid_size') is not None:
        document += '候选轴展开共 ' + str(evidence['compatible_grid_size']) + ' 组，没有穷举全部交互。'
    document += '</p><ul>'
    for item in evidence['skipped']:
        document += '<li>' + escape(item['option'] + '：' + item['reason']) + '</li>'
    document += '<li>本页基准也是选定的量化模型；不是关闭全部优化的原始BF16模型。</li><li>若量化和加速阶段使用不同数据，不能把两阶段的加速比相乘。</li></ul>'
    document += '<details id="raw-records"><summary>查看全部初筛记录、实际内核与默认参数</summary><div class="table-wrap"><table><thead><tr><th>阶段 / 配置</th><th>实际矩阵内核</th><th>状态</th><th>cold E2E秒</th><th>warm E2E秒</th></tr></thead><tbody>'
    for record in evidence['records']:
        for result in record['results'] or [{}]:
            cold, warm = (result.get('cache', {}).get(cache, {}) for cache in ('cold', 'warm'))
            target = duplicate_target(record)
            status = '完整' if record['complete'] else '执行合并，参见 ' + target if target else '未完成'
            document += '<tr><td>' + escape(record['phase'] + ' / ' + record['label']) + '</td><td>' + escape(', '.join((record['observed_kernels'] or {}).keys()) or '未记录') + '</td><td>' + escape(status) + '</td><td>' + number(cold.get('e2e_s')) + '</td><td>' + number(warm.get('e2e_s')) + '</td></tr>'
    document += '</tbody></table></div><h3>记录的共同设置</h3>'
    common = evidence['recorded_common_settings_by_profile']
    distinct = {json.dumps(value, sort_keys=True, ensure_ascii=False) for value in common.values()}
    data = json.loads(next(iter(distinct))) if len(distinct) == 1 else common
    document += '<pre>' + escape(json.dumps(data, ensure_ascii=False, indent=2)) + '</pre><p><a href="search-details.json">下载完整筛选数据</a></p></details>'
    return document + PAGE_END
