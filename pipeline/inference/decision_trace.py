"""Explain a staged search using only measurements available at each stage."""
import math
import statistics


SETTING_KEYS = ('attention', 'kernel', 'compile', 'graph', 'budget', 'mtp')
STAGES = {
    'attention': {'title': '选注意力后端', 'axis': ('attention',),
                  'purpose': '加快模型读取上下文的注意力计算；FA2、FA3、FlashInfer是不同实现。'},
    'kernels': {'title': '选量化矩阵内核', 'axis': ('kernel',),
                'purpose': '加快量化模型的矩阵乘法；auto由框架选择，Humming和Marlin是可选实现。'},
    'graphs': {'title': '选编译与CUDA Graph', 'axis': ('compile', 'graph'),
               'purpose': '编译优化计算过程，CUDA Graph复用已录制的GPU执行步骤，减少重复调度开销。',
               'options_help': '无Graph：逐次调度；分段Graph：分块录制与重放；完整＋分段Graph：在支持的执行条件下使用完整图，其余使用分段图。'},
    'budgets': {'title': '选调度预算', 'axis': ('budget',),
                'purpose': '设置每轮调度最多处理多少token，在处理效率和等待时间间取得平衡；不是输出长度。'},
    'mtp': {'title': '选MTP预测步数', 'axis': ('mtp',),
            'purpose': '让草稿模型先预测后续token，再由主模型验证；预测越准，越有机会加快生成。',
            'options_help': 'MTP1/MTP2分别表示一次先预测1/2个token；预测更多也可能增加验证开销，因此需要实测。'},
}


def option_label(stage, settings):
    if stage == 'attention':
        return {'fa2': 'FA2', 'fa3': 'FA3', 'flashinfer': 'FlashInfer'}.get(settings['attention'], settings['attention'])
    if stage == 'kernels':
        return {'auto': 'auto（框架自动选）', 'humming': 'Humming', 'marlin': 'Marlin'}.get(settings['kernel'], settings['kernel'])
    if stage == 'graphs':
        if not settings['compile']:
            return 'Eager（不编译、不录图）'
        return {'none': '编译＋无Graph', 'piecewise': '编译＋分段Graph',
                'full': '编译＋完整＋分段Graph'}.get(settings['graph'], str(settings['graph']))
    if stage == 'budgets':
        return str(settings['budget'])
    return 'MTP关闭' if not settings['mtp'] else 'MTP' + str(settings['mtp'])


def fixed_label(settings, changed_stage):
    return ' · '.join(option_label(stage, settings) for stage in STAGES if stage != changed_stage)


def duplicate_target(record):
    results = record.get('results', [])
    if not results or any(result.get('stage2_status') != 'duplicate' for result in results):
        return None
    targets = {result.get('duplicate_of') for result in results}
    return next(iter(targets)) if len(targets) == 1 and None not in targets and '' not in targets else None


def _stats(record, cache):
    results = record.get('results', [])
    if not record.get('complete') or not results:
        return None
    measures = [result.get('cache', {}).get(cache, {}) for result in results]
    if any(not isinstance(item.get('e2e_s'), (int, float)) or not math.isfinite(item['e2e_s'])
           or item['e2e_s'] <= 0 for item in measures):
        return None
    def mean(key):
        values = [item.get(key) for item in measures]
        return statistics.mean(values) if all(isinstance(value, (int, float)) and math.isfinite(value)
                                             for value in values) else None
    return {key: mean(key) for key in ('e2e_s', 'ttft_s', 'decode_tps', 'quality_score')}


def _option_order(stage, settings):
    if stage == 'attention':
        return {'fa2': 0, 'fa3': 1, 'flashinfer': 2}.get(settings['attention'], 99)
    if stage == 'kernels':
        return {'auto': 0, 'marlin': 1, 'humming': 2}.get(settings['kernel'], 99)
    if stage == 'graphs':
        return 0 if not settings['compile'] else {'none': 1, 'piecewise': 2, 'full': 3}.get(settings['graph'], 99)
    return settings['budget' if stage == 'budgets' else 'mtp']


def build(evidence, config):
    """Reconstruct comparisons; never reinterpret final results as early decisions."""
    if config['search'].get('strategy') != 'staged':
        return []
    objective = config['search']['objective_cache']
    keep = config['search'].get('finalists', 2)
    baseline_id = (evidence.get('baseline') or {}).get('id')
    cumulative, steps = {}, []
    for phase in evidence['phases']:
        name = phase['phase']
        added = [record for record in evidence['records'] if record['phase'] == name]
        cumulative.update({record['profile']: record for record in added})
        if name not in STAGES:
            continue
        # Comparison cohorts must match; old small-sample scores cannot stand
        # in for a differently sized later experiment.
        comparable = [record for record in cumulative.values()
                      if record['samples_per_cache'] == phase['samples_per_cache']
                      and record['expected_repeats'] == phase['repeats']]
        metrics = {record['profile']: _stats(record, objective) for record in comparable}
        baseline = metrics.get(baseline_id)
        reference = baseline.get('quality_score') if baseline else None
        def qualified(record):
            item = metrics[record['profile']]
            return item is not None and (reference is None or (item['quality_score'] is not None
                                                              and item['quality_score'] + 1e-12 >= reference))
        ranked = sorted((record for record in comparable if qualified(record)),
                        key=lambda record: (metrics[record['profile']]['e2e_s'], record['profile']))
        retained = ranked[:keep]
        axis = STAGES[name]['axis']
        fixed_keys = [key for key in SETTING_KEYS if key not in axis]
        def context(record):
            return tuple(record['settings'][key] for key in fixed_keys)
        contexts = list(dict.fromkeys(context(record) for record in added))
        leader = ranked[0] if ranked else None
        if leader and context(leader) not in contexts:
            contexts.insert(0, context(leader))
        groups = []
        for fixed in contexts:
            members = sorted((record for record in comparable if context(record) == fixed),
                             key=lambda record: (_option_order(name, record['settings']), record['profile']))
            if not members:
                continue
            choices = [record for record in ranked if context(record) == fixed]
            best_id = choices[0]['profile'] if choices else None
            gap = None
            if len(choices) > 1:
                fastest, next_best = (metrics[record['profile']]['e2e_s'] for record in choices[:2])
                gap = (next_best - fastest) / fastest
            groups.append({'fixed': dict(zip(fixed_keys, fixed)),
                           'fixed_label': fixed_label(members[0]['settings'], name),
                           'primary': bool(leader and context(leader) == fixed), 'best_id': best_id,
                           'relative_gap': gap,
                           'rows': [{'profile': record['profile'], 'option': option_label(name, record['settings']),
                                     'settings': record['settings'], 'source_phase': record['phase'],
                                     'reused': record['phase'] != name, 'complete': record['complete'],
                                     'duplicate_of': duplicate_target(record),
                                     'duplicate_label': cumulative.get(duplicate_target(record), {}).get('label'),
                                     'quality_eligible': qualified(record),
                                     'retained': any(record['profile'] == value['profile'] for value in retained),
                                     'cold': _stats(record, 'cold'), 'warm': _stats(record, 'warm'),
                                     'observed_kernels': record.get('observed_kernels'),
                                     'mtp_acceptance': record.get('mtp_acceptance')}
                                    for record in members]})
        groups.sort(key=lambda group: not group['primary'])
        steps.append({'phase': name, 'title': STAGES[name]['title'], 'purpose': STAGES[name]['purpose'],
                      'options_help': STAGES[name].get('options_help'),
                      'skipped': [item for item in evidence.get('skipped', [])
                                  if item['option'] in {'attention': ('fa2', 'fa3', 'flashinfer'),
                                                        'kernels': ('auto', 'marlin', 'humming'),
                                                        'mtp': ('mtp',)}.get(name, ())],
                      'samples_per_cache': phase['samples_per_cache'], 'repeats': phase['repeats'],
                      'quality_reference': reference,
                      'objective_cache': objective, 'groups': groups,
                      'leader': {'profile': leader['profile'], 'settings': leader['settings'],
                                 'option': option_label(name, leader['settings'])} if leader else None,
                      'retained': [{'profile': record['profile'], 'settings': record['settings'],
                                    'label': record['label'], 'option': option_label(name, record['settings'])}
                                   for record in retained]})
    return steps
