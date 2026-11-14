"""Simple public comparison with an optional detailed engineering report."""
import csv
import html
import json
from pathlib import Path
import statistics
from .common import read_json, write_json
from .benchmark import percentile


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def average(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def stats(rows):
    good = [row for row in rows if row['status'] in ('success','fixed_complete')]
    result = {'requests':len(rows),'successes':len(good),
              'errors':sum(row['status']=='error' for row in rows),
              'truncated':sum(row['status']=='truncated' for row in rows)}
    for key in ['e2e_s','ttft_s','decode_tps','tpot_ms','input_tokens','output_tokens']:
        result[key+'_mean']=average(good,key)
        result[key+'_p50']=percentile([row.get(key) for row in good],0.5)
        result[key+'_p95']=percentile([row.get(key) for row in good],0.95)
    return result


def key_of(row):
    return row['uid'],row['concurrency'],row['cache']


def format_value(value):
    if value is None: return '—'
    return f'{value:.3f}' if isinstance(value,float) else str(value)


def table(records,columns):
    result='<table><thead><tr>'+''.join('<th>'+html.escape(title)+'</th>' for _,title in columns)+'</tr></thead><tbody>'
    for record in records:
        result+='<tr>'+''.join('<td>'+html.escape(format_value(record.get(key)))+'</td>' for key,_ in columns)+'</tr>'
    return result+'</tbody></table>'


def curve(series, metric, title):
    colors=['#1d4ed8','#b45309','#047857','#9333ea','#374151']
    maximum=max([row[metric] for rows in series.values() for row in rows if row.get(metric) is not None] or [1])
    count=max([len(rows) for rows in series.values()] or [2])
    svg=f'<h3>{html.escape(title)}</h3><svg viewBox="0 0 940 300" role="img" aria-label="{html.escape(title)}">'
    for index in range(5):
        y=240-index*50
        svg+=f'<line x1="70" y1="{y}" x2="920" y2="{y}" stroke="#ddd"/><text x="5" y="{y+5}">{maximum*index/4:.2f}</text>'
    for index,(name,rows) in enumerate(series.items()):
        points=[]
        for position,row in enumerate(rows):
            if row.get(metric) is not None:
                points.append(f'{70+position*850/max(count-1,1):.2f},{240-row[metric]/maximum*200:.2f}')
        color=colors[index%len(colors)]
        svg+=f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.5"/>'
        svg+=f'<text x="{70+index*230}" y="280" fill="{color}">{html.escape(name)}</text>'
    return svg+'</svg><p>横轴：同一批样本的固定顺序；同一列对应同一个请求。曲线仅展示所有已完成方案共同正常结束的样本。</p>'


def profile_label(profile):
    if profile.get('label'):
        return profile['label']
    parts = [{'fa2': 'FA2', 'fa3': 'FA3', 'flashinfer': 'FlashInfer'}[profile['attention']]]
    if profile.get('mtp'):
        parts.append('MTP' + str(profile['mtp']))
    parts.append('CUDA Graph' if profile.get('graph', True) else 'Eager')
    return ' + '.join(parts)


def comparison_point(config):
    settings = config.get('report', {})
    cache_modes = config['benchmark'].get('cache_modes', ['cold'])
    concurrency = settings.get('concurrency', 1)
    cache = settings.get('cache', 'warm' if 'warm' in cache_modes else 'cold')
    if type(concurrency) is not int or concurrency < 1 or cache not in cache_modes:
        raise ValueError('报告对比点必须为正整数并发和本轮已配置的缓存状态')
    return concurrency, cache


def ranked_candidates(overview, concurrency, cache, expected_requests):
    candidates = [row for row in overview if row['state'] == 'completed'
                  and row['concurrency'] == concurrency and row['cache'] == cache
                  and row['requests'] == expected_requests and expected_requests > 0
                  and row['successes'] == expected_requests
                  and row['errors'] == 0 and row['truncated'] == 0
                  and row['common_requests'] == expected_requests
                  and row['common_e2e_s'] is not None]
    return sorted(candidates, key=lambda row: row['common_e2e_s'])


def combined_summary_table(summary):
    cache_modes = summary['cache_modes']
    metrics = [('ttft_s_mean', 'TTFT（秒）↓'), ('e2e_s_mean', 'E2E（秒）↓'),
               ('decode_tps_mean', 'TPS（token/s）↑')]
    records = {}
    for row in summary['rows']:
        record = records.setdefault(row['profile'], {'label': row['label'], 'caches': {}})
        record['caches'][row['cache']] = row
    document = '<table><thead><tr><th rowspan="2">加速方案</th>'
    for cache in cache_modes:
        label = 'cold：首次前缀' if cache == 'cold' else 'warm：重复前缀'
        document += '<th colspan="3" class="cache-group">' + label + '</th>'
    document += '</tr><tr>'
    for _ in cache_modes:
        for index, (_, label) in enumerate(metrics):
            boundary = ' class="group-start"' if index == 0 else ''
            document += '<th' + boundary + '>' + label + '</th>'
    document += '</tr></thead><tbody>'
    for record in records.values():
        document += '<tr><td>' + html.escape(record['label']) + '</td>'
        for cache in cache_modes:
            values = record['caches'].get(cache, {})
            for index, (key, _) in enumerate(metrics):
                boundary = ' class="group-start"' if index == 0 else ''
                document += '<td' + boundary + '>' + html.escape(format_value(values.get(key))) + '</td>'
        document += '</tr>'
    return document + '</tbody></table>'


def acceleration_context(run_dir, profiles, statuses):
    completed = [profile for profile in profiles if statuses[profile['id']]['status'] == 'completed']
    common = None
    recorded_budgets = []
    for profile in completed:
        directory = run_dir / 'profiles' / profile['id']
        launch_path = directory / 'launch.json'
        if not launch_path.is_file():
            common = set()
            break
        command = read_json(launch_path)['command']
        audit_path = directory / 'loading_audit.json'
        audit = read_json(audit_path) if audit_path.is_file() else {}
        def argument(name, default=None):
            return command[command.index(name) + 1] if name in command else default
        compilation = json.loads(argument('--compilation-config', '{}'))
        recorded_budgets.append(argument('--max-num-batched-tokens'))
        audit_compilation = audit.get('compilation', '')
        features = set()
        if (argument('--quantization') == 'awq_marlin' and audit.get('passed')
                and any('Marlin' in name for name in audit.get('methods', {}))):
            features.add('awq_marlin')
        if (compilation.get('mode') == 3 and '--enforce-eager' not in command
                and 'VLLM_COMPILE' in audit_compilation):
            features.add('compile')
        if (compilation.get('cudagraph_mode', 'NONE') != 'NONE' and '--enforce-eager' not in command
                and any(mode in audit_compilation for mode in ['CUDAGraphMode.FULL', 'CUDAGraphMode.PIECEWISE'])):
            features.add('graph')
        if audit.get('passed') and audit.get('attention_implementations'):
            features.add('kv_cache')
        if '--enable-prefix-caching' in command and '--no-enable-prefix-caching' not in command:
            features.add('prefix_cache')
        if '--enable-chunked-prefill' in command and '--no-enable-chunked-prefill' not in command:
            features.add('chunked_prefill')
        common = features if common is None else common & features
    labels = [('awq_marlin', 'AWQ4 + Marlin'), ('compile', 'torch.compile 编译优化'),
              ('graph', 'CUDA Graph'), ('kv_cache', 'KV Cache（引擎基础机制）'),
              ('prefix_cache', '前缀缓存'), ('chunked_prefill', '分块预填充')]
    variations = []
    attention = list(dict.fromkeys(profile['attention'] for profile in completed))
    if len(attention) > 1:
        names = {'fa2': 'FA2', 'fa3': 'FA3', 'flashinfer': 'FlashInfer'}
        variations.append('注意力后端（' + ' / '.join(names[name] for name in attention) + '）')
    mtp = list(dict.fromkeys(profile.get('mtp', 0) for profile in completed))
    if len(mtp) > 1:
        variations.append('MTP（' + ' / '.join('MTP' + str(value) if value else '关闭' for value in mtp) + '）')
    for key, default, label in [('graph', True, 'CUDA Graph 开关'), ('prefix_cache', True, '前缀缓存开关'),
                                ('batched_tokens', 8192, '调度 token 预算')]:
        if len({profile.get(key, default) for profile in completed}) > 1:
            variations.append(label)
    fixed = []
    if common and 'awq_marlin' in common:
        fixed.append('量化矩阵内核：AWQ Marlin')
    if (len(recorded_budgets)==len(completed) and recorded_budgets
            and all(value is not None for value in recorded_budgets) and len(set(recorded_budgets))==1):
        fixed.append('分块预填充调度预算：'+recorded_budgets[0]+' tokens')
    return {'common_features': [{'id': key, 'label': label} for key, label in labels if key in (common or set())],
            'fixed_settings':fixed,'comparison_dimensions': variations}


def simple_document(config, manifest, doctor, result):
    concurrency = result['summary']['concurrency']
    profiles = {profile['id']: profile for profile in config['profiles']}
    source = ((manifest.get('quantization_provenance') or {}).get('recipe', {}).get('source', {}).get('path')
              or config['quantization'].get('source') or manifest.get('source_model', '当前模型'))
    model = Path(source).name
    escape = html.escape
    document = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>推理加速简表</title><style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#20252b;line-height:1.7}h1{font-size:26px}table{width:100%;border-collapse:collapse;margin:20px 0;font-size:14px}th,td{padding:12px;border-bottom:1px solid #ddd;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}thead tr:first-child th:first-child,tbody td:first-child{text-align:left}th{background:#f5f5f5}th.cache-group{text-align:center;border-left:1px solid #ddd}.group-start{border-left:1px solid #ddd}small,.meta{color:#59616b}.table-wrap{overflow-x:auto}strong{font-weight:650}</style></head><body><h1>推理加速对比</h1>'''
    document += f'<p class="meta">{escape(model)} · {escape(str(config["quantization"]["method"]).upper())} · {escape(doctor["gpu"])}<br>{result["summary"]["samples"]} 条通用数据 · 并发 {concurrency} · 以下均为平均值</p>'
    context = result['summary']['acceleration_context']
    if context['common_features']:
        document += '<p><strong>各方案共同开启：</strong>' + escape(' · '.join(item['label'] for item in context['common_features'])) + '。</p>'
    else:
        document += '<p class="meta">共同加速配置：本轮记录不足，未列出。</p>'
    if context['comparison_dimensions']:
        document += '<p><strong>本次主要对比：</strong>' + escape('；'.join(context['comparison_dimensions'])) + '。下表列出已完成实测的候选组合。</p>'
    else:
        document += '<p class="meta">下表列出本轮已完成实测的候选组合。</p>'
    if context['fixed_settings']:
        document += '<p class="meta"><strong>本轮固定：</strong>' + escape('；'.join(context['fixed_settings'])) + '。</p>'
    document += '<div class="table-wrap">' + combined_summary_table(result['summary']) + '</div>'
    recommendations = {}
    unavailable = []
    for cache in result['summary']['cache_modes']:
        ranked = ranked_candidates(result['overview'], concurrency, cache, result['summary']['samples'])
        if ranked:
            recommendations.setdefault(ranked[0]['profile'], []).append(cache)
        else:
            unavailable.append(cache)
    for name, caches in recommendations.items():
        document += '<p><strong>' + ' / '.join(caches) + ' 推荐：' + escape(profile_label(profiles[name])) + '</strong>（平均 E2E 最低）</p>'
    if unavailable:
        document += '<p>' + ' / '.join(unavailable) + ' 暂无满足完整测试条件的推荐方案。</p>'
    notes = []
    for name, status in result['statuses'].items():
        if status['status'] == 'completed':
            if not any(row['profile'] == name and row['complete'] for row in result['summary']['rows']):
                notes.append(profile_label(profiles[name]) + '：缺少完整的本次对比数据')
        else:
            label = '未测试' if status['status'] == 'skipped' else '测试未完成'
            if profiles[name]['attention'] == 'fa3' and status['status'] == 'skipped':
                label = '当前 GPU 不支持，已跳过'
            notes.append(profile_label(profiles[name]) + '：' + label)
    if notes:
        document += '<p class="meta">' + escape('；'.join(notes)) + '。</p>'
    document += '<p>TTFT：等到首个输出的时间；E2E：整段回复的总耗时；TPS：生成速度。时间越低越好，TPS 越高越好。</p>'
    document += '<p><small>前缀缓存通常保持开启。cold 表示本次前缀尚未缓存，warm 表示重复请求可复用已有前缀；两种测试都已完成内核预热。TPS 按流式输出估算。此表用于当前模型和 GPU 的通用配置初筛，业务效果与实际负载请另测。</small></p>'
    document += '</body></html>'
    return document


def generate(run_dir, detailed=False):
    config=read_json(run_dir/'config.json')
    profiles=config['profiles']
    rows_by_profile={}
    statuses={}
    for profile in profiles:
        directory=run_dir/'profiles'/profile['id']
        statuses[profile['id']]=read_json(directory/'status.json') if (directory/'status.json').exists() else {'status':'not_run'}
        rows_by_profile[profile['id']]=read_rows(directory/'requests.jsonl')
    completed=[name for name,status in statuses.items() if status['status']=='completed']
    sets=[{key_of(row) for row in rows_by_profile[name] if row['status'] in ('success','fixed_complete')} for name in completed]
    common=set.intersection(*sets) if sets else set()
    overview=[]
    buckets=[]
    for name,rows in rows_by_profile.items():
        for concurrency in config['benchmark']['concurrencies']:
            for cache in config['benchmark'].get('cache_modes',['cold']):
                selected=[row for row in rows if row['concurrency']==concurrency and row['cache']==cache]
                paired=[row for row in selected if key_of(row) in common]
                batches=[b for b in read_rows(run_dir/'profiles'/name/'batches.jsonl')
                         if b['concurrency']==concurrency and b['cache']==cache]
                seconds=sum(b['wall_s'] for b in batches)
                item={'profile':name,'state':statuses[name]['status'],'concurrency':concurrency,'cache':cache,
                      **stats(selected),'common_requests':len(paired),'common_e2e_s':average(paired,'e2e_s'),
                      'request_qpm':60*len(selected)/seconds if seconds else None,
                      'output_throughput_tps':sum(r.get('output_tokens',0) for r in selected)/seconds if seconds else None}
                overview.append(item)
                for group in dict.fromkeys(row['group'] for row in rows):
                    buckets.append({'profile':name,'concurrency':concurrency,'cache':cache,'group':group,
                                    **stats([r for r in paired if r['group']==group])})
    # Rank only a full, error-free, non-truncated cohort at one declared operating point.
    target_concurrency, target_cache = comparison_point(config)
    expected_requests = len(read_json(run_dir/'inputs.json'))
    ranked = ranked_candidates(overview, target_concurrency, target_cache, expected_requests)
    recommendation={'profile':ranked[0]['profile'] if ranked else None,
        'basis':f'并发{target_concurrency}、{target_cache} 前缀缓存、共同完整样本的平均E2E最低',
        'quality_validated':False,'requires_user_selection':True,
        'limits':'性能推荐仅适用于本次模型、GPU和数据；未运行质量评分；不兼容方案及原因见任务状态。'}
    write_json(run_dir/'recommendation.json',recommendation)
    summary_rows = []
    for row in overview:
        if row['concurrency'] != target_concurrency or row['state'] != 'completed':
            continue
        profile = next(profile for profile in profiles if profile['id'] == row['profile'])
        complete = row['requests'] == expected_requests and row['successes'] == expected_requests and expected_requests > 0
        summary_rows.append({'profile': row['profile'], 'label': profile_label(profile), 'cache': row['cache'],
                             'complete': complete, **{key: row[key] if complete else None for key in
                                 ['ttft_s_mean', 'e2e_s_mean', 'decode_tps_mean']}})
    report={'overview':overview,'buckets':buckets,'statuses':statuses,'recommendation':recommendation,
            'summary':{'concurrency':target_concurrency,'cache':target_cache,
                       'cache_modes':config['benchmark'].get('cache_modes',['cold']),
                       'samples':expected_requests,'rows':summary_rows,
                       'acceleration_context':acceleration_context(run_dir,profiles,statuses)}}
    write_json(run_dir/'report.json',report)
    for filename,records in [('overview.csv',overview),('buckets.csv',buckets)]:
        if records:
            with (run_dir/filename).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    with (run_dir/'summary.csv').open('w',newline='') as stream:
        columns=['label','cache','ttft_s_mean','e2e_s_mean','decode_tps_mean']
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(summary_rows)
    manifest=read_json(run_dir/'model_manifest.json')
    doctor=read_json(run_dir/'doctor.json')
    (run_dir/'report.html').write_text(simple_document(config,manifest,doctor,report))
    if not detailed:
        return report
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>推理加速对比</title><style>body{font:15px system-ui,sans-serif;max-width:1400px;margin:30px auto;padding:0 20px;color:#17202a}table{border-collapse:collapse;width:100%;margin:16px 0;font-size:13px}th,td{padding:9px;border:1px solid #d7dce1;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#f3f5f7}svg{width:100%;max-width:1100px;font:12px system-ui}p{line-height:1.7}pre{white-space:pre-wrap}h2{margin-top:36px}</style><h1>推理加速实测</h1>'''
    document+=f'<p>模型模式：{html.escape(manifest["mode"])}；GPU：{html.escape(doctor["gpu"])}；vLLM：{doctor["vllm"]}。结果来自本轮真实 API 请求。</p>'
    document+='<p>TTFT：首个生成增量（含推理或工具）；E2E：发请求到收到 [DONE]；单请求TPS=(输出token−1)/(末次增量−首次增量)。MTP/SSE可能一次合并多个token，该TPS是客户端估计。P95是分位数：耗时越低越好，TPS越高越好。QPM按各批请求的实际墙钟时间计算，不包含启动、预热和清缓存开销。</p>'
    document+='<p>冷缓存：每批测量前重置前缀缓存；热缓存：原批原样再次请求。它们都已完成内核预热，不代表首次冷启动。自然生成的截断单列，不参与正常完成耗时。未进行业务质量评价。</p><h2>总表</h2>'
    document+=table(overview,[('profile','方案'),('state','状态'),('concurrency','并发'),('cache','缓存'),('requests','请求'),('successes','完整'),('errors','失败'),('truncated','截断'),('e2e_s_mean','E2E均值秒'),('e2e_s_p95','E2E P95'),('ttft_s_mean','TTFT均值秒'),('ttft_s_p95','TTFT P95'),('decode_tps_mean','TPS均值'),('tpot_ms_mean','TPOT毫秒'),('request_qpm','QPM'),('output_throughput_tps','总输出tokens/s')])
    for metric,label in [('e2e_s','E2E / 秒'),('ttft_s','TTFT / 秒'),('decode_tps','单请求 TPS')]:
        document+=f'<h2>{label}：任务分桶</h2>'+table(buckets,[('profile','方案'),('concurrency','并发'),('cache','缓存'),('group','任务类型'),('requests','共同样本数'),(metric+'_mean','均值'),(metric+'_p50','P50'),(metric+'_p95','P95')])
        for concurrency in config['benchmark']['concurrencies']:
            for cache in config['benchmark'].get('cache_modes',['cold']):
                series={name:sorted([r for r in rows_by_profile[name] if key_of(r) in common and r['concurrency']==concurrency and r['cache']==cache],key=lambda r:r['uid']) for name in completed}
                document+=curve(series,metric,f'{label} · 并发{concurrency} · {cache}')
    document+='<h2>推荐与边界</h2><pre>'+html.escape(json.dumps(recommendation,ensure_ascii=False,indent=2))+'</pre>'
    document+='<h2>组合状态</h2><pre>'+html.escape(json.dumps(statuses,ensure_ascii=False,indent=2))+'</pre></html>'
    (run_dir/'report-detailed.html').write_text(document)
    return report
