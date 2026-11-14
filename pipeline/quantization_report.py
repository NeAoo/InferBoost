"""Stage-one report: paired quality gates, complete cohorts, explicit model choice."""
import csv
import html
import json
import math
from pathlib import Path
import re
import statistics

from . import quality
from .common import read_json, write_json
from .quantization_labels import (ALGORITHM_EXPLANATIONS, PRECISION_NOTE,
                                  describe_quantization, quantization_sort_key)
from .report_layout import PAGE_END, page_start


def mean(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def resource_metrics(directory, manifest):
    """Historical observations; file size, model allocation and cache reservation differ."""
    directory = Path(directory)
    log = directory / 'server.log'
    text = log.read_text(errors='replace') if log.is_file() else ''
    def last_measurement(pattern):
        values = re.findall(pattern, text)
        return float(values[-1]) if values else None
    samples = directory / 'gpu.jsonl'
    gpu_memory = []
    if samples.is_file():
        for line in samples.read_text().splitlines():
            sample = json.loads(line)
            columns = next(csv.reader([sample.get('sample', '')]))
            if sample.get('returncode') != 0 or len(columns) != 6:
                continue
            try:
                value = float(columns[2]) / 1024
            except ValueError:
                continue
            if math.isfinite(value) and value >= 0:
                gpu_memory.append(value)
    memory = directory / 'quantized/memory_usage.jsonl'
    if not memory.is_file() and manifest.get('artifact'):
        memory = Path(manifest['artifact']) / 'memory_usage.jsonl'
    snapshots = [json.loads(line) for line in memory.read_text().splitlines()] if memory.is_file() else []
    rss = [row['VmHWM_bytes'] / 1024**3 for row in snapshots if row.get('VmHWM_bytes') is not None]
    return {'model_load_gpu_gib': last_measurement(r'Model loading took ([0-9.]+) GiB memory'),
            'service_gpu_sample_peak_gib': max(gpu_memory) if gpu_memory else None,
            'kv_cache_budget_gib': last_measurement(r'Available KV cache memory: ([0-9.]+) GiB'),
            'quantization_cpu_peak_gib': max(rss) if rss else None}


def generate(run_dir, persist=True):
    run_dir = Path(run_dir)
    config = read_json(run_dir / 'config.json')
    original = next(item['id'] for item in config['candidates'] if item['recipe'] == 'original')
    statuses, evaluation, overview = {}, {}, []
    for candidate in config['candidates']:
        name = candidate['id']
        directory = run_dir / 'candidates' / name
        path = directory / 'status.json'
        statuses[name] = read_json(path) if path.exists() else {'status': 'not_run'}
        evaluation[name] = {}
        manifest = read_json(directory / 'model_manifest.json') if (directory / 'model_manifest.json').exists() else {}
        resources = resource_metrics(directory, manifest)
        for dataset in config['datasets']:
            rows = read_json(run_dir / 'datasets' / (dataset['id'] + '.json'))
            dataset_dir = directory / 'datasets' / dataset['id']
            separate = config.get('benchmark', {}).get('separate_quality', False)
            path = dataset_dir / ('quality/requests.jsonl' if separate else 'requests.jsonl')
            attempts = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            primary = [row for row in attempts if row['concurrency'] == 1 and row['cache'] == 'cold']
            result = quality.summarize(rows, primary)
            result.update(e2e_s=mean([row for row in primary if row['status'] == 'success'], 'e2e_s'),
                          ttft_s=mean([row for row in primary if row['status'] == 'success'], 'ttft_s'),
                          tps=mean([row for row in primary if row['status'] == 'success'], 'decode_tps'))
            performance = {'repeats': 1, 'complete': result['complete']}
            if separate:
                from .measurement import performance_summary
                performance = performance_summary(dataset_dir, rows, config['benchmark'])
                result.update({key: performance[key] for key in ['e2e_s', 'ttft_s', 'tps']})
                result['quality_complete'] = result['complete']
                result['complete'] &= performance['complete']
            result['performance'] = performance
            evaluation[name][dataset['id']] = result
            overview.append({'candidate': name, 'recipe': candidate['recipe'], 'dataset': dataset['id'],
                             'state': statuses[name]['status'], 'requests': len(primary), 'expected_requests': len(rows),
                             'complete': result['complete'], 'scored_count': result['scored_count'],
                             'score_pct': result['score_pct'], 'e2e_s': result['e2e_s'], 'ttft_s': result['ttft_s'],
                             'tps': result['tps'], 'weight_gib': sum(s['bytes'] for s in manifest.get('identity', {}).get('shards', {}).values()) / 1024**3 if manifest else None,
                             'performance_repeats': performance['repeats'],
                             'e2e_s_std': performance.get('e2e_s_std'), 'ttft_s_std': performance.get('ttft_s_std'),
                             'tps_std': performance.get('tps_std'),
                             **resources, 'mtp_unchanged': manifest.get('mtp_unchanged'), 'drop_pp': None, 'regressions': None})
    gates = {}
    for candidate in config['candidates']:
        name = candidate['id']
        conditions = []
        passed = statuses[name]['status'] == 'completed'
        has_scored = False
        for dataset in config['datasets']:
            dataset_id = dataset['id']
            baseline, current = evaluation[original][dataset_id], evaluation[name][dataset_id]
            # Every compared operating point must contain the exact frozen cohort.
            complete = baseline['complete'] and current['complete']
            indexed = {row['uid']: row for row in current['details']}
            regressions = [row['uid'] for row in baseline['details'] if row.get('score') == 1 and indexed[row['uid']].get('score') == 0] if complete else []
            scored = baseline['score_pct'] is not None and current['score_pct'] is not None and baseline['scored_count'] == current['scored_count']
            drop = baseline['score_pct'] - current['score_pct'] if scored else None
            threshold = config['quality_gate']
            quality_ok = scored and drop <= threshold.get('max_drop_pp', 0) and len(regressions) <= threshold.get('max_regressions', 0)
            has_scored |= scored
            passed &= complete and (quality_ok if baseline['scored_count'] else True)
            conditions.append({'dataset': dataset_id, 'complete': complete, 'scored': scored,
                               'drop_pp': drop, 'regressions': regressions, 'quality_passed': quality_ok if scored else None})
            for row in overview:
                if row['candidate'] == name and row['dataset'] == dataset_id:
                    row.update(drop_pp=drop, regressions=len(regressions) if scored else None)
        gates[name] = {'passed': bool(passed and has_scored), 'datasets': conditions,
                       'scope': 'relative_to_original_on_declared_rule_tests', 'business_quality_validated': False}
    selectable = {name: statuses[name]['status'] == 'completed' and all(
        result['complete'] for result in evaluation[name].values()) for name in statuses}
    for row in overview:
        row['quality_gate_passed'] = gates[row['candidate']]['passed'] if row['complete'] else None
        row['selectable'] = selectable[row['candidate']]
    primary_dataset = config['datasets'][0]['id']
    objective = config.get('selection', {}).get('objective', 'latency')
    if objective not in ('latency', 'size', 'quality'):
        raise ValueError('模型选择目标仅支持latency、size或quality')
    candidates = [row for row in overview if row['dataset'] == primary_dataset and gates[row['candidate']]['passed']]
    key = 'weight_gib' if objective == 'size' else 'e2e_s'
    candidates = sorted((row for row in candidates if row[key] is not None), key=lambda row: row[key])
    if objective == 'quality':
        candidates.sort(key=lambda row: (-row['score_pct'], row['e2e_s']))
    recommendation = {'candidate': candidates[0]['candidate'] if candidates else None, 'objective': objective,
                      'basis': '所有数据完整，带答案数据满足相对原模型质量门槛；在首份数据C1冷前缀下按目标排序',
                      'requires_user_selection': True, 'production_validated': False,
                      'quality_criteria_are_advisory': True,
                      'unlabeled_datasets': [dataset['id'] for dataset in config['datasets'] if evaluation[original][dataset['id']]['scored_count'] == 0]}
    result = {'overview': overview, 'statuses': statuses, 'candidate_gates': gates, 'selectable': selectable,
              'recommendation': recommendation, 'evaluation': evaluation,
              'experience': config.get('experience', {}),
              'measurement_note': ('独立确定性评分1轮；性能' + str(config['benchmark']['performance_repeats'])
                 + '轮，各轮先对同一批样本求均值，再对轮次等权平均；±为轮均值的样本标准差。重复不增加独立题目数。'
                 if config.get('benchmark', {}).get('separate_quality') else '历史协议：评分与性能来自同一轮请求。')}
    return render(run_dir, config, result) if persist else result


def render(run_dir, config, result):
    """Render recorded measurements; do not rerun or recalculate an old experiment."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    overview, statuses = result['overview'], result['statuses']
    gates, evaluation = result['candidate_gates'], result['evaluation']
    recommendation = result['recommendation']
    result['progress_state'] = ('complete' if all(item['status'] in ('completed', 'failed', 'skipped')
                                                for item in statuses.values()) else 'in_progress')
    original = next(item['id'] for item in config['candidates'] if item['recipe'] == 'original')
    for row in overview:
        row.update(describe_quantization(row['recipe']))
    overview.sort(key=quantization_sort_key)
    result['precision_note'] = PRECISION_NOTE
    # Use the frozen run configuration, never today's defaults for old reports.
    result['quality_gate'] = dict(config['quality_gate'])
    write_json(run_dir / 'report.json', result)
    write_json(run_dir / 'recommendation.json', recommendation)
    with (run_dir / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(overview[0]))
        writer.writeheader()
        writer.writerows(overview)
    escape = lambda value: html.escape(str(value))
    display = lambda value: '—' if value is None else f'{value:.3f}' if isinstance(value, float) else str(value)
    columns = [('candidate', '模型候选ID'), ('quantization_label', '算法＋精度'),
               ('weight_precision', '权重精度'), ('activation_precision', '激活输入精度'),
               ('recipe', '量化配方ID'), ('dataset', '数据集'), ('state', '状态'),
               ('requests', '请求数'), ('score_pct', '规则题得分%'), ('quality_gate_passed', '符合推荐质量条件'), ('selectable', '可手动选择'), ('drop_pp', '下降百分点'),
               ('regressions', '原对现错题数'), ('e2e_s', 'E2E秒'), ('ttft_s', 'TTFT秒'),
               ('tps', '单请求TPS估计'), ('weight_gib', '权重文件GiB'),
               ('model_load_gpu_gib', '模型加载显存GiB'), ('service_gpu_sample_peak_gib', '服务采样显存峰值GiB'),
               ('kv_cache_budget_gib', 'KV缓存预算GiB'), ('quantization_cpu_peak_gib', '量化CPU内存峰值GiB'),
               ('mtp_unchanged', 'MTP保持不变')]
    document = page_start('第一步：选一个适合的量化模型')
    document += '<p class="lead">先看答题表现，再权衡模型大小、显存和速度；推荐供参考，最终由你选择。</p>'
    from .text_report import quantization_gate_note
    document += '<p class="notice">' + escape(quantization_gate_note(result)) + '</p>'
    document += '<p>' + escape(result.get('measurement_note', '')) + '</p>'
    if result.get('experience', {}).get('smoke_only'):
        document += '<p class="notice">快速体验：仅验证链路，样本与搜索范围缩减，不构成正式部署推荐。</p>'
    document += '<p><a href="report.md">Markdown版（可在终端阅读）</a></p>'
    if result['progress_state'] == 'in_progress':
        document += '<p class="notice">仍在执行：以下为当前已完成结果与暂定建议，最终结果可能改变。</p>'
    document += '<h2>先看懂这些名字</h2><dl>'
    for name, explanation in ALGORITHM_EXPLANATIONS:
        document += '<dt>' + escape(name) + '</dt><dd>' + escape(explanation) + '</dd>'
    document += '</dl><p class="caption">默认仅压缩权重，激活保持16位（本流程为BF16）。INT8/FP8激活会单独标明；位数更少不保证推理更快或质量更好。MTP不参与量化。</p>'
    document += '<h2>量化结果对比</h2><p class="meta">排序：原模型对照 → GPTQ → AWQ → AWQ＋GPTQ → 其他；同算法内按W4A16 → W8A16 → W8A8排列。排序不代表性能排名。</p>'
    document += '<p class="caption">并发1 · 冷前缀 · 内核已预热 · 相同推理配置 · MTP关闭。时间越低越好，TPS越高越好。得分来自本次规则题，不代表业务准确率；无标准答案时显示“—”。</p>'
    main_columns = [('score_pct', '本测试集得分%'), ('weight_gib', '权重GiB'),
                    ('model_load_gpu_gib', '加载显存GiB'), ('ttft_s', '首字秒'),
                    ('e2e_s', '总耗时秒'), ('tps', 'TPS估计')]
    for dataset in config['datasets']:
        if len(config['datasets']) > 1:
            document += '<h3>数据集：' + escape(dataset['id']) + '</h3>'
        document += '<div class="table-wrap"><table><thead><tr><th>方案 / 激活输入</th>' + ''.join('<th>' + title + '</th>' for _, title in main_columns) + '<th>选择提示</th></tr></thead><tbody>'
        for row in overview:
            if row['dataset'] != dataset['id']:
                continue
            suggested = row['candidate'] == recommendation['candidate']
            if suggested:
                choice = '本轮推荐'
            elif row.get('selectable'):
                degraded = (row.get('drop_pp') or 0) > 0 or (row.get('regressions') or 0) > 0
                choice = '可选 · 有质量退化' if degraded else '可选 · 未评分' if row.get('score_pct') is None else '可选'
            else:
                choice = {'skipped': '已跳过', 'failed': '执行失败', 'not_run': '尚未测试'}.get(row['state'], '尚未完成')
            document += '<tr' + (' class="winner"' if suggested else '') + '><td>' + escape(row['quantization_label']) + '<span class="state-note">' + escape(row['activation_precision']) + '</span></td>'
            for key, _ in main_columns:
                value = display(row[key])
                if row.get(key + '_std') is not None:
                    value += ' ± ' + display(row[key + '_std'])
                if key == 'score_pct' and row[key] is not None:
                    value = f'{row[key]:.3f}'.rstrip('0').rstrip('.')
                document += '<td>' + escape(value) + '</td>'
            document += '<td>' + escape(choice) + '</td></tr>'
        document += '</tbody></table></div>'
    recommended_row = next((row for row in overview if row['candidate'] == recommendation['candidate']), None)
    if recommended_row:
        objective = {'size': '权重文件更小', 'latency': '平均总耗时更低',
                     'quality': '本测试集得分更高，同分时耗时更低'}[recommendation['objective']]
        document += '<p class="decision"><strong>本轮建议：' + escape(recommended_row['quantization_label']) + '</strong>。在符合推荐质量条件的候选中，按' + objective + '选择；你仍可选择其他已完成评测的模型。</p>'
    else:
        document += '<p>当前没有符合完整性与质量条件的推荐模型；失败或未完成的候选不可选择。</p>'
    document += '<p>选择后会生成 <code>selected_model.json</code>，携带模型路径、量化配方和校验信息，交给第二步测试推理加速。</p>'
    document += '<details><summary>查看完整资源指标、精度说明和执行记录</summary>'
    document += '<p>' + escape(PRECISION_NOTE) + '</p>'
    document += '<p>模型加载显存来自vLLM日志；服务显存峰值是5秒采样的GPU总占用，包含KV Cache与运行开销。量化CPU峰值是量化进程RSS高水位。“MTP保持不变”通过张量dtype、形状和SHA256检查。缺失数值显示“—”。</p>'
    document += '<div class="table-wrap"><table><thead><tr>' + ''.join('<th>' + title + '</th>' for _, title in columns) + '</tr></thead><tbody>'
    for row in overview:
        document += '<tr>' + ''.join('<td>' + escape(display(row[key])) + '</td>' for key, _ in columns) + '</tr>'
    document += '</tbody></table></div><h3>推荐条件</h3><pre>' + escape(json.dumps(config['quality_gate'], ensure_ascii=False, indent=2)) + '</pre><h3>执行状态</h3><pre>' + escape(json.dumps(statuses, ensure_ascii=False, indent=2)) + '</pre></details>'
    document += '<details><summary>查看逐题退化：哪些题原模型答对、量化后答错</summary>'
    for name, gate in gates.items():
        if name == original:
            continue
        for condition in gate['datasets']:
            dataset_id = condition['dataset']
            before = {row['uid']: row for row in evaluation[original][dataset_id]['details']}
            after = {row['uid']: row for row in evaluation[name][dataset_id]['details']}
            document += '<h3>' + escape(name + ' / ' + dataset_id) + '</h3>'
            if not condition['complete']:
                document += '<p>本条件尚未完成同批请求，无法判断是否新增退化题。</p>'
            elif not condition['scored']:
                document += '<p>没有可配对的标准答案得分，不判断量化质量变化。</p>'
            elif not condition['regressions']:
                document += '<p>本次带答案题中未发现原模型答对、此候选答错的情况。</p>'
            for uid in condition['regressions']:
                document += '<details><summary>' + escape(uid) + '</summary><pre>' + escape(json.dumps({'expected': before[uid]['expected'], 'original': before[uid]['content'], 'candidate': after[uid]['content']}, ensure_ascii=False, indent=2)) + '</pre></details>'
    document += '</details>' + PAGE_END
    (run_dir / 'report.html').write_text(document)
    from .text_report import quantization_markdown
    (run_dir / 'report.md').write_text(quantization_markdown(result))
    return result
