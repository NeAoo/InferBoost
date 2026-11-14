"""Guided configuration, reusing the workflow's public command arguments."""
from pathlib import Path
import shlex
import time

from . import jobs, preflight, ui
from .common import identifier, write_json
from .launcher import model_input, path_input, positive_integer, normalize_gpus
from .modes import MODES
from .terminal import flow_path

RECIPE_HELP = {'gptq4': '4位权重，BF16激活', 'awq4': '激活感知的4位权重量化',
               'gptq8': '8位权重，BF16激活', 'w8a8': 'INT8权重与动态INT8激活',
               'awq_gptq4': 'AWQ平滑后GPTQ量化', 'w4afp8': '4位权重与FP8激活',
               'fp8': 'FP8权重与动态FP8激活'}


def preferences(root):
    return jobs.read_optional(Path(root) / '.inferboost-ui.json')


def remember(root, model):
    data = preferences(root)
    data['recent_models'] = list(dict.fromkeys([str(model), *data.get('recent_models', [])]))[:8]
    write_json(Path(root) / '.inferboost-ui.json', data)


def resource_panel(resources):
    storage, memory = resources['storage'], resources['memory']
    ui.key_values('环境与工作目录', [('工作目录', storage['path']),
        ('磁盘可用', f"{storage['free_gib']:.1f} GiB"), ('持久性', storage['persistence']),
        ('容器内存上限', f"{memory['limit_gib']:.1f} GiB" if memory['limit_gib'] is not None else '未知'),
        ('GPU', '；'.join(f"{g['index']}: {g['name']} {g['total_gib']:.0f} GiB" for g in resources['gpus']) or '未检测到')])
    ui.console.print('这是资源检查；真实GPU加载与推理在任务中验收。', style='dim')


def budget_fields(skip_quantization):
    quantization = [('calibration_count', '校准条数（内置最多512）'),
                    ('calibration_tokens', '校准token上限'), ('performance_repeats', '量化评测性能轮数')]
    acceleration = [('screening_samples', '加速初筛条数'), ('search_trials', '加速初筛组合预算'),
                    ('final_repeats', '加速最终复测轮数')]
    return acceleration if skip_quantization else quantization + acceleration


def planned_configs(root, state, skip_quantization):
    from .workflow import build_configs, parse_arguments
    return build_configs(root, parse_arguments(['--root', str(root), *build_arguments(state, skip_quantization)]))


def measurement_rows(state, skip_quantization, quantization, inference):
    rows = [('GPU', state['gpus']), ('编译并发', state['max_jobs']), ('上下文', state['max_model_len'])]
    if not skip_quantization:
        rows += [('校准', f"{quantization['calibration']['count']}条"),
                 ('量化评测', f"质量独立1次；性能{quantization['benchmark']['performance_repeats']}轮")]
    search = inference['search']
    rows += [('加速初筛', f"每个组合最多{search['screening_samples']}条；最多{search['max_trials']}个组合"),
             ('加速最终复测', f"{search['repeats']}轮；cold/warm分开，预热不计入"),
             ('质量复核', '有标准答案时独立核验；无答案仅测性能')]
    return rows


def guide(root, skip=False):
    from .inference.defaults import defaults_text
    root = Path(root)
    resources = preflight.inspect_resources(root, check_write=True)
    resource_panel(resources)
    ui.console.print(defaults_text(), markup=False)
    state = {'mode': 'standard', 'model': '', 'mtp': '', 'dataset': '', 'objective': 'size',
             'gpus': str(resources['gpus'][0]['index']) if resources['gpus'] else '0',
             'max_jobs': '2', 'max_model_len': '8192', 'advanced': {},
             'run_id': 'forge-' + time.strftime('%Y%m%d-%H%M%S'), 'auto_select': False}
    steps = [('model', '模型'), ('mtp', 'MTP'), ('mode', '运行模式')]
    if not skip:
        steps.append(('recipes', '量化配方'))
    steps += [('dataset', '评测数据'), ('resources', '资源与预算'),
              ('selection', '方案选择' if skip else '选择偏好'), ('review', '运行确认')]
    step_keys = [key for key, _ in steps]
    index = 0
    while True:
        step, label = steps[index]
        ui.heading(f'配置 {index + 1}/{len(steps)} · {label}', flow_path(skip))
        try:
            if step == 'model':
                recent = preferences(root).get('recent_models', [])
                if recent:
                    ui.console.print('最近模型：\n' + '\n'.join(recent), markup=False, style='dim')
                state['model'] = str(model_input(ui.text_input('模型目录', state['model'], path=True), skip))
                model_path = Path(state['model'])
                if model_path.is_file():
                    model_path = Path(jobs.read_optional(model_path)['model'])
                state['summary'] = preflight.model_summary(model_path)
                model = state['summary']
                ui.key_values('已识别模型', [('架构', model['model_type']), ('配置dtype', model['precision']),
                    ('权重', f"{model['weight_gib']:.2f} GiB / {model['shards']} 个分片"),
                    ('已有量化', '是' if model['quantized'] else '否'), ('发现MTP张量', model['mtp_tensors'])])
                if model['quantized']:
                    ui.console.print('配置dtype不是量化权重位宽；实际权重/激活格式以量化配置和加载核验为准。', style='dim')
                if skip:
                    ui.console.print('本次复用模型：只做加速对比、质量复核与部署选择。', style='cyan')
                remember(root, state['model'])
                if model['max_model_len']:
                    state['max_model_len'] = str(min(int(state['max_model_len']), model['max_model_len']))
            elif step == 'mtp':
                value = ui.text_input('MTP目录（无则留空；same=主模型目录）', state['mtp'], path=True, optional=True)
                if value == 'same':
                    value = state['summary']['path']
                if value and not (path_input(value) / 'config.json').is_file():
                    raise ValueError('MTP目录缺少config.json；提供的草稿还需在运行时核验匹配。')
                state['mtp'] = str(path_input(value)) if value else ''
            elif step == 'mode':
                previous = state['mode']
                state['mode'] = ui.choose('本次运行模式', [(key, value['label'] + ' · ' + value['scope'], None)
                                         for key, value in MODES.items()], default=state['mode'])
                if state['mode'] != previous:
                    state.pop('recipes', None)
                    state['advanced'] = {}
            elif step == 'recipes':
                selected_gpus = [g for g in resources['gpus'] if str(g['index']) in state['gpus'].split(',')]
                choices = preflight.recipe_choices(state['summary'], selected_gpus)
                defaults = state.get('recipes', MODES[state['mode']]['recipes'])
                state['recipes'] = ui.choose('量化方法 · A16表示BF16激活',
                    [(row['id'], row['label'] + ' · ' + RECIPE_HELP[row['id']], row['reason']) for row in choices],
                    default=defaults, multiple=True)
                ui.console.print('兼容性为静态准入，本模型的实际质量和速度仍需测量。', style='dim')
            elif step == 'dataset':
                ui.console.print('默认加速性能集 public100（100条），质量复核集 quality100（100条），用途不同。', style='dim')
                if not skip:
                    ui.console.print('量化候选先使用 quality100 做质量与速度对比。', style='dim')
                ui.console.print('自定义数据用于加速负载；带答案的样本做质量复核，无答案仅测性能。快速体验只取部分样本。', style='dim')
                value = ui.text_input('自定义测试JSON/JSONL（留空使用上述默认数据）', state['dataset'], path=True, optional=True)
                if value:
                    from .quality import load_dataset
                    dataset = load_dataset(path_input(value))
                    ui.console.print(f'已读取{len(dataset)}条；有标准答案{sum("expected" in row for row in dataset)}条。')
                state['dataset'] = str(path_input(value)) if value else ''
            elif step == 'resources':
                quantization, inference = planned_configs(root, state, skip)
                ui.key_values('资源与测量设置', measurement_rows(state, skip, quantization, inference))
                action = ui.choose('资源设置', [('keep', '使用以上设置', None), ('edit', '修改GPU、上下文与编译并发', None),
                    ('budget', '修改加速初筛和复测预算' if skip else '修改校准、重复次数和搜索预算',
                     None if state['mode'] == 'custom' else '切换到自定义模式后可改')])
                if action == 'edit':
                    devices = normalize_gpus(ui.text_input('推理GPU编号（单卡）' if skip else 'GPU编号，逗号分隔', state['gpus']))
                    if skip and ',' in devices:
                        raise ValueError('直接加速当前使用单卡推理，请填写一个GPU编号')
                    state['gpus'] = devices
                    state['max_jobs'] = str(positive_integer(ui.text_input('编译并发（不是内存上限）', state['max_jobs'])))
                    state['max_model_len'] = str(positive_integer(ui.text_input('上下文上限', state['max_model_len'])))
                elif action == 'budget':
                    actual = {'calibration_count': quantization['calibration']['count'],
                              'calibration_tokens': quantization['calibration']['max_tokens'],
                              'performance_repeats': quantization['benchmark']['performance_repeats'],
                              'screening_samples': inference['search']['screening_samples'],
                              'search_trials': inference['search']['max_trials'],
                              'final_repeats': inference['search']['repeats']}
                    proposed = dict(state['advanced'])
                    for key, label in budget_fields(skip):
                        proposed[key] = positive_integer(ui.text_input(label, actual[key]))
                    planned_configs(root, dict(state, advanced=proposed), skip)
                    state['advanced'] = proposed
            elif step == 'selection':
                if not skip:
                    state['objective'] = ui.choose('量化模型推荐偏好（最终仍由你决定）',
                        [('size', '资源优先：质量达标后优先较小权重', None),
                         ('latency', '延迟优先：质量达标后优先较低E2E', None),
                         ('quality', '质量优先：本测试集得分优先，同分再看E2E', None)], default=state['objective'])
                _, inference = planned_configs(root, state, skip)
                cache = inference['search']['objective_cache']
                ui.console.print(f'加速推荐：并发1、{cache}前缀缓存状态，完整复测平均E2E最低；有答案时需通过质量复核。TTFT、TPS另列供比较。', style='dim')
                state['auto_select'] = ui.choose('加速完成后如何选择部署方案' if skip else '量化模型与加速方案两个选择点如何处理',
                    [('manual', '等待我查看报告并选择', None), ('auto', '按加速推荐继续导出并验证部署包' if skip else '授权按本次推荐自动继续', None)],
                    default='auto' if state['auto_select'] else 'manual') == 'auto'
            else:
                state['run_id'] = identifier(ui.text_input('任务名称', state['run_id']))
                arguments = build_arguments(state, skip)
                # Validate all overrides before launching or freezing a run.
                quantization, inference = planned_configs(root, state, skip)
                resources = preflight.inspect_resources(root, check_write=True)
                devices = list(map(int, state['gpus'].split(',')))
                review = preflight.validate_resources(resources, devices, state['summary'], state.get('recipes', []), skip)
                model_limit = state['summary'].get('max_model_len')
                if model_limit and int(state['max_model_len']) > model_limit:
                    review['errors'].append(f'上下文超过模型声明的{model_limit}，请返回修改资源。')
                if not skip:
                    choices = preflight.recipe_choices(state['summary'], [g for g in resources['gpus'] if g['index'] in devices])
                    review['errors'] += [row['id'] + '：' + row['reason'] for row in choices if row['id'] in state['recipes'] and row['reason']]
                for issue in review['errors']:
                    ui.console.print(issue, style='red', markup=False)
                for warning in review['warnings']:
                    ui.console.print(warning, style='yellow', markup=False)
                ui.key_values('确认运行计划', [('模型', state['model']), ('MTP', state['mtp'] or '关闭'),
                    ('模式', MODES[state['mode']]['label']),
                    ('模型处理', '复用现有模型，直接测试加速') if skip else ('量化配方', ', '.join(state['recipes'])),
                    ('方案选择', '按推荐自动导出并验证部署包' if state['auto_select'] else '报告完成后等待手动选择'),
                    ('产物目录', root / 'runs'), ('内存与磁盘', '资源预估不保证不会OOM；运行中持续显示占用'),
                    ('测试范围', MODES[state['mode']]['scope'])])
                ui.key_values('本次资源与测量预算', measurement_rows(state, skip, quantization, inference))
                ui.console.print(command_preview(root, arguments), markup=False, style='dim')
                action = ui.choose('下一步', [('start', '开始后台任务并打开看板', '请先修正上方错误' if review['errors'] else None),
                    ('model', '返回修改模型', None), ('mode', '返回修改运行模式' if skip else '返回修改模式与配方', None),
                    ('resource', '返回修改资源', None), ('quit', '返回主菜单', None)])
                if action == 'start':
                    return jobs.submit(root, arguments)
                if action == 'quit':
                    raise ui.Cancel()
                index = step_keys.index({'model': 'model', 'mode': 'mode', 'resource': 'resources'}[action])
                continue
            index += 1
        except ui.Back:
            index = max(0, index - 1)
        except (ValueError, OSError, KeyError) as error:
            ui.console.print(str(error), style='red', markup=False)


def command_preview(root, arguments):
    return '等价命令：' + shlex.join([str(root / 'quickstart'), 'workflow', *arguments])


def build_arguments(state, skip):
    arguments = ['--model', state['model'], '--mode', state['mode'], '--run-id', state['run_id'],
                 '--gpus', state['gpus'],
                 '--max-jobs', state['max_jobs'], '--max-model-len', state['max_model_len']]
    if skip:
        arguments.append('--skip-quantization')
    else:
        arguments += ['--selection-objective', state['objective'], '--recipes', ','.join(state['recipes'])]
    for field, option in [('mtp', '--mtp-model'), ('dataset', '--dataset')]:
        if state[field]:
            arguments += [option, state[field]]
    if state['auto_select']:
        arguments.append('--auto-select')
    allowed = {key for key, _ in budget_fields(skip)}
    for key, value in state['advanced'].items():
        if key not in allowed:
            continue
        arguments += ['--' + key.replace('_', '-'), str(value)]
    return arguments


def history(root):
    from .dashboard import watch
    base = Path(root) / 'jobs'
    entries = sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True) if base.is_dir() else []
    if not entries:
        ui.console.print('尚无后台任务；旧版任务可从纯文本菜单查看。')
        return
    value = ui.choose('选择任务 · 可输入名字搜索',
        [(p.name, p.name + ' · ' + jobs.status(root, p.name)['state'], None) for p in entries[:40]])
    watch(root, value)


def menu(root):
    from .launcher import deploy, doctor, history as legacy_history
    from .dashboard import watch
    from .product import NAME, CHINESE_NAME, VERSION
    from .terminal import LOGO
    if ui.console.width >= 60:
        ui.console.print(LOGO, style='bold cyan', markup=False)
    ui.heading(f'{NAME} · {CHINESE_NAME}  {VERSION}', '量化、推理加速、测试对比与部署')
    resource_panel(preflight.inspect_resources(root))
    while True:
        try:
            action = ui.choose('开始工作', [('new', '新模型 · 量化 → 评测 → 加速 → 部署', None),
                ('quantized', '已有量化模型 · 直接加速', None), ('history', '任务中心 · 续看 / 选择 / 续跑', None),
                ('retest', '从旧工作目录重新选模型 · 复用量化结果', None),
                ('legacy', '查看原有任务与报告', None), ('deploy', '启动已有部署包', None),
                ('workspace', '工作目录 · 更换存储路径', None),
                ('doctor', '环境自检与存储说明', None), ('exit', '退出', None)])
            if action == 'exit':
                return
            if action in ('new', 'quantized'):
                run_id = guide(root, action == 'quantized')
                watch(root, run_id)
            elif action == 'history':
                history(root)
            elif action == 'retest':
                from .dashboard import choose_retest
                directory = ui.text_input('原任务目录（工作目录/runs/workflows/任务名）', path=True)
                run_id = choose_retest(root, source_directory=directory)
                if run_id:
                    watch(root, run_id)
            elif action == 'legacy':
                legacy_history(root)
            elif action == 'deploy':
                deploy(root)
            elif action == 'workspace':
                from .storage import change_workspace
                change_workspace(root)
            elif action == 'doctor':
                doctor(root)
        except (ui.Back, ui.Cancel):
            ui.console.print('已返回菜单；后台任务保持原状态。', style='dim')
        except (EOFError, KeyboardInterrupt):
            ui.console.print('已退出菜单；已提交的后台任务保持原状态。', style='dim')
            return
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            ui.console.print(str(error), style='red', markup=False)
