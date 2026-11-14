"""Plain Markdown summaries shared by saved reports and terminal output."""
from .quantization_labels import ALGORITHM_EXPLANATIONS, PRECISION_NOTE


def number(value, places=4):
    return '—' if value is None else f'{value:.{places}f}'


def cell(value):
    return str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '\\|').replace('\n', ' ')


def table(headers, rows):
    lines = ['| ' + ' | '.join(map(cell, headers)) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |']
    lines.extend('| ' + ' | '.join(map(cell, row)) + ' |' for row in rows)
    return '\n'.join(lines) + '\n'


def quantization_gate_note(result):
    gate = result.get('quality_gate')
    if gate is None:
        return '本次推荐质量门槛未记录在报告中；以任务 config.json 为准。'
    return (f"本次推荐质量门槛：每份带答案数据的总分下降 ≤ {gate.get('max_drop_pp', 0):g} 分（百分制），"
            f"且原模型答对、量化后答错 ≤ {gate.get('max_regressions', 0)} 道。"
            '两项同时满足；量化和全部评测须成功、完整。此门槛只用于推荐，最终由用户选择。')


def quantization_summary(result, candidate=None):
    rows = [row for row in result['overview'] if candidate is None or row['candidate'] == candidate]
    output = table(['候选ID / 算法精度', '数据', '状态', '本测试集得分%', '权重GiB', '加载显存GiB',
                    'TTFT秒', 'E2E秒', 'TPS', '可选择'],
                   [[row['candidate'] + ' / ' + row['quantization_label'], row['dataset'], row['state'],
                     number(row['score_pct'], 2), number(row['weight_gib'], 2), number(row.get('model_load_gpu_gib'), 2),
                     number(row['ttft_s']), number(row['e2e_s']), number(row['tps'], 2),
                     '是' if row.get('selectable') else '否'] for row in rows])
    provisional = result.get('progress_state') == 'in_progress'
    recommendation = result['recommendation']['candidate'] or '暂无'
    return quantization_gate_note(result) + '\n\n' + output + '\n' + ('进行中，暂定建议' if provisional else '本轮建议') + '：' + recommendation + '；最终由用户选择。\n'


def quantization_markdown(result):
    document = '# 第一步：量化对比\n\n'
    document += '并发1、冷前缀、内核已预热、MTP关闭。得分为本批规则题得分，不代表业务准确率。\n\n'
    document += result.get('measurement_note', '') + '\n\n'
    if result.get('experience', {}).get('smoke_only'):
        document += '**快速体验，仅验证链路，不作为正式部署推荐。**\n\n'
    document += quantization_summary(result)
    if any(row.get('performance_repeats', 1) > 1 for row in result['overview']):
        document += '\n轮间波动（轮均值的样本标准差，非请求级P95）：\n\n'
        document += table(['候选', '数据', '轮数', 'TTFT标准差秒', 'E2E标准差秒', 'TPS标准差'],
            [[row['candidate'], row['dataset'], row.get('performance_repeats'), number(row.get('ttft_s_std')),
              number(row.get('e2e_s_std')), number(row.get('tps_std'))] for row in result['overview']])
    document += '\n## 名称说明\n\n'
    document += '\n'.join('- **' + name + '**：' + explanation for name, explanation in ALGORITHM_EXPLANATIONS)
    document += '\n\n' + PRECISION_NOTE + '\n\n'
    document += '加载显存、服务显存峰值、KV缓存预算是不同指标。全部资源与逐题退化见 [HTML](report.html)；完整数据见 [JSON](report.json)。\n\n'
    for name, state in result['statuses'].items():
        if state.get('reason') or state.get('error'):
            document += '- ' + cell(name + '：' + state.get('reason', state.get('error', ''))) + '\n'
    document += '\n选择后生成 `selected_model.json`，将模型路径与校验信息交给第二步。\n'
    return document


def report_paths(directory, search=False):
    names = ['report.md', 'report.html']
    if search:
        names += ['search-details.md', 'search-details.html']
    print('\n'.join('报告：' + str(directory / name) for name in names), flush=True)
