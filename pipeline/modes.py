"""Explicit experiment budgets shared by CLI and guided UI."""
MODES = {
    'quick': {'label': '快速体验', 'scope': '链路验证，不能作为正式质量或部署推荐',
              'recipes': ['gptq4'], 'calibration_count': 64, 'calibration_tokens': 1024,
              'sample_limit': 10, 'performance_repeats': 1, 'screening_samples': 5,
              'search_trials': 8, 'final_repeats': 1},
    'standard': {'label': '标准对比', 'scope': '当前模型、GPU与本测试集的完整对比',
                 'recipes': ['gptq4', 'awq4', 'gptq8', 'w8a8'], 'calibration_count': 512,
                 'calibration_tokens': 4096, 'sample_limit': None, 'performance_repeats': 3,
                 'screening_samples': 20, 'search_trials': 24, 'final_repeats': 3},
    'custom': {'label': '自定义', 'scope': '用户自定义范围，详见冻结参数',
               'recipes': ['gptq4', 'awq4', 'gptq8', 'awq_gptq4', 'w8a8', 'w4afp8', 'fp8'],
               'calibration_count': 512, 'calibration_tokens': 4096, 'sample_limit': None,
               'performance_repeats': 3, 'screening_samples': 20, 'search_trials': 24, 'final_repeats': 3},
}


def apply_mode(quantization, inference, args):
    mode = getattr(args, 'mode', 'custom')
    if mode not in MODES:
        raise ValueError('未知运行模式')
    values = dict(MODES[mode])
    if mode == 'custom':
        values.update(calibration_count=quantization['calibration'].get('count', 512),
                      calibration_tokens=quantization['calibration'].get('max_tokens', 4096),
                      performance_repeats=quantization['benchmark'].get('performance_repeats', 3),
                      screening_samples=inference['search'].get('screening_samples', 20),
                      search_trials=inference['search'].get('max_trials', 24),
                      final_repeats=inference['search'].get('repeats', 3))
    limits = {'calibration_count': 512, 'calibration_tokens': 32768,
              'performance_repeats': 20, 'screening_samples': 10000, 'search_trials': 1000,
              'final_repeats': 20}
    for key, maximum in limits.items():
        override = getattr(args, key, None)
        if override is not None:
            if type(override) is not int or not 1 <= override <= maximum:
                raise ValueError(f'{key} 必须在1–{maximum}范围内')
            values[key] = override
    quantization['calibration'].update(count=values['calibration_count'], max_tokens=values['calibration_tokens'])
    quantization['benchmark'].update(performance_repeats=values['performance_repeats'],
                                    separate_quality=True, sample_limit=values['sample_limit'])
    objective = getattr(args, 'selection_objective', 'size')
    if objective not in ('quality', 'latency', 'size'):
        raise ValueError('选择偏好必须是quality、latency或size')
    quantization.setdefault('selection', {})['objective'] = objective
    inference['search'].update(screening_samples=values['screening_samples'],
                               max_trials=values['search_trials'], repeats=values['final_repeats'])
    inference['benchmark']['sample_limit'] = values['sample_limit']
    inference['quality']['sample_limit'] = values['sample_limit']
    for config in (quantization, inference):
        config['experience'] = {'mode': mode, 'label': values['label'], 'scope': values['scope'],
                                'smoke_only': mode == 'quick'}
    return values


def sample_rows(rows, limit):
    if limit is None or len(rows) <= limit:
        return rows
    # Take a deterministic round-robin over groups, rather than the first task only.
    groups = {}
    for row in rows:
        groups.setdefault(row['group'], []).append(row)
    selected, offset = [], 0
    while len(selected) < limit:
        for group in groups.values():
            if offset < len(group):
                selected.append(group[offset])
                if len(selected) == limit:
                    return selected
        offset += 1
    return selected
