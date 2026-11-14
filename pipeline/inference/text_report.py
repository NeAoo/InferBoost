"""Stage summaries from the same evidence as HTML; usable directly over SSH."""
from pipeline.text_report import number, table
from . import mtp_metrics
from .defaults import defaults_markdown


def stage_markdown(step):
    lines = [f'## {step["title"]}', '', step['purpose'], '',
             f'每种缓存 {step["samples_per_cache"]} 条 × {step["repeats"]} 轮；目标：{step["objective_cache"]}平均E2E最低。', '']
    for item in step.get('skipped', []):
        lines += [f'跳过 {item["option"]}：{item["reason"]}。', '']
    for group in step['groups']:
        lines += ['固定条件：' + group['fixed_label'], '']
        headers = ['本步选项', 'cold E2E秒', 'warm E2E秒', 'warm TTFT秒', 'warm TPS']
        if step['phase'] == 'mtp':
            headers.append('草稿token接受率')
        headers += ['结论', '来源']
        rows = []
        for row in group['rows']:
            cold, warm = row['cold'] or {}, row['warm'] or {}
            conclusion = ('实际执行相同，已合并' if row.get('duplicate_of') else
                          '未完成' if not row['complete'] else '质量或数据未达标' if not row['quality_eligible']
                          else '本组均值最低' if row['profile'] == group['best_id'] else '备选' if row['retained'] else '耗时较高')
            values = [row['option'], number(cold.get('e2e_s')), number(warm.get('e2e_s')),
                      number(warm.get('ttft_s')), number(warm.get('decode_tps'), 2)]
            if step['phase'] == 'mtp':
                values.append(mtp_metrics.label(row.get('mtp_acceptance')))
            source = ('参见 ' + (row.get('duplicate_label') or row['duplicate_of']) if row.get('duplicate_of')
                      else '复用 ' + row['source_phase'] if row['reused'] else '本步实测')
            rows.append(values + [conclusion, source])
        lines += [table(headers, rows)]
    leader = step['leader']
    lines += ['本步领先：' + (leader['option'] if leader else '暂无有效结论') + '。',
              '保留配置：' + '；'.join(item['label'] for item in step['retained']) + '。', '']
    if step['phase'] == 'mtp':
        lines += [mtp_metrics.EXPLANATION, '标记“性能合计”的新实验：' + mtp_metrics.PERFORMANCE_SCOPE + '。',
                  '历史实验：' + mtp_metrics.LOG_SCOPE + '。', '']
    return '\n'.join(lines)


def acceptance_table(acceptance, candidates):
    if not any(item.get('status') != 'disabled' for item in acceptance.values()):
        return ''
    rows = []
    for profile, item in acceptance.items():
        rows.append([candidates.get(profile, {}).get('label', profile), mtp_metrics.label(item),
                     item.get('accepted_tokens', '—'), item.get('drafted_tokens', '—'),
                     item.get('scope', item.get('reason', '未采集'))])
    return '\n### MTP草稿token接受率\n\n' + table(['方案', '接受率', '接受token', '草稿token', '统计范围'], rows) + '\n' + mtp_metrics.EXPLANATION + '\n'


def final_markdown(report, candidates):
    rows = report['rows']
    document = '# 第二步：完整复测结果\n\n' + defaults_markdown() + '并发1；cold/warm分开统计；以下为均值。初筛不混入最终排名。\n\n'
    if report.get('experience', {}).get('smoke_only'):
        document += '**快速体验，仅验证链路；缩减样本和搜索范围，不作为正式部署推荐。**\n\n'
    document += table(['方案ID / 配置', '缓存', '请求数', 'TTFT秒', 'E2E秒', 'TPS', '质量条件'],
                      [[row['profile'] + ' / ' + row['label'], row['cache'], row['requests'], number(row['ttft_s_mean']),
                        number(row['e2e_s_mean']), number(row['decode_tps_mean'], 2),
                        '通过' if report['quality_gates'].get(row['profile']) and report.get('quality_scored')
                        else '未评分' if not report.get('quality_scored') else '未通过'] for row in rows])
    recommended = report.get('recommendation')
    document += '\n本轮建议：' + candidates.get(recommended, {}).get('label', '暂无完整有效推荐') + '。最终由用户选择。\n'
    document += acceptance_table(report.get('mtp_acceptance', {}), candidates)
    document += '\n默认开启KV Cache、前缀缓存、分块预填充，KV精度auto；未对这些默认项单独消融。基准也是所选量化模型，不是关闭全部优化的BF16原模型。\n'
    document += '\n[五步依据](search-details.md) · [HTML](report.html) · [完整JSON](report.json)\n'
    return document


def search_markdown(evidence, model_label):
    document = '# 第二步：五步加速选型\n\n' + defaults_markdown() + '模型：' + model_label + '\n\n'
    document += '注意力 → 矩阵内核 → 编译与Graph → 调度预算 → MTP → 完整复测 → 用户选择。\n\n'
    document += '每一步保留少数组合继续测试；早期领先值可能在后续组合测试中改变，不保证全局最优。\n\n'
    document += '\n'.join(stage_markdown(step) for step in evidence.get('decision_steps', []))
    if not evidence.get('decision_steps'):
        document += '尚无可重建的分阶段记录；不把完整网格或固定配置验证描述为五步筛选。\n'
    final = evidence.get('final_validation', {})
    recommended = final.get('recommended_trial')
    document += '\n## 最终结论\n\n' + ('完整复测推荐：' + recommended['label'] if recommended else '仍在筛选，尚无最终推荐。') + '\n'
    document += '\n[完整复测](report.md) · [交互式HTML](search-details.html) · [全部筛选数据](search-details.json)\n'
    document += '\n默认KV Cache、前缀缓存、分块预填充已开启，未单独比较开关。cold/warm均已预热。\n'
    if evidence.get('skipped'):
        document += '\n本轮跳过：\n\n' + '\n'.join('- ' + item['option'] + '：' + item['reason'] for item in evidence['skipped']) + '\n'
    return document
