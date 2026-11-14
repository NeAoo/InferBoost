"""A read-only live view; leaving this application never signals the worker."""
import asyncio
import io
from pathlib import Path
import re
import time

from . import jobs, preflight

STATES = {'starting': '正在启动', 'running': '运行中', 'stopping': '停止中，正在清理服务',
          'completed': '完整流程通过', 'failed': '失败', 'interrupted': '已中断',
          'awaiting_model_selection': '等待选择量化模型', 'awaiting_profile_selection': '等待选择加速配置'}

LOG_TAIL_BYTES = 256 * 1024
MAX_LOG_LINES = 50
REFRESH_SECONDS = 1
RESOURCE_REFRESH_SECONDS = 5


def choose_retest(root, run_id=None, source_directory=None):
    from . import ui
    from .common import identifier
    from .retest import source
    directory, quant, _ = source(source_directory or Path(root) / 'runs/workflows' / run_id)
    options = ui.show_report(quant.parents[1], quant.name.removesuffix('-quant'))
    candidate = ui.choose('返回模型选择 · 复用权重，新建加速对比', options)
    default = directory.name[:40] + '-' + candidate[:16] + '-' + time.strftime('%m%d-%H%M%S')
    name = identifier(ui.text_input('新的任务名称（保留原任务）', default))
    if any(path.exists() for path in [Path(root) / 'runs/workflows' / name,
                                     Path(root) / 'runs/inference' / (name + '-accel')]):
        raise ValueError('任务名称已存在，请选择新名称')
    choice = ui.choose('沿用原数据、MTP和搜索预算；重新测量加速与质量', [
        ('start', '开始新对比；完成后由我选择部署方案', None), ('back', '返回，不运行', None)])
    if choice == 'back':
        return None
    return jobs.submit(root, ['--from-workflow', str(directory), '--candidate', candidate, '--run-id', name])


def service_preparation(candidate):
    """Read service lifecycle markers without probing or signalling its process."""
    process = jobs.read_optional(candidate / 'process.json')
    started = process.get('started_at')
    if not started:
        return ''
    cleanup = jobs.read_optional(candidate / 'cleanup.json')
    if cleanup.get('finished_at', 0) >= started:
        return ''
    ready = candidate / 'ready.json'
    if ready.is_file() and ready.stat().st_mtime >= started:
        return '服务就绪，评测预热中'
    return '启动 vLLM：加载 / 编译 / 预热'


def snapshot(root, run_id):
    root = Path(root)
    job = jobs.status(root, run_id)
    request = jobs.read_optional(root / 'runs/workflows' / run_id / 'request.json')
    arguments = jobs.read_optional(root / 'jobs' / run_id / 'request.json').get('arguments', [])
    skip_quantization = request.get('skip_quantization', '--skip-quantization' in arguments or '--from-workflow' in arguments)
    can_reselect = not skip_quantization or bool(request.get('retest_source') or '--from-workflow' in arguments)
    quantization = root / 'runs' / (run_id + '-quant')
    counts = {'completed': 0, 'failed': 0, 'skipped': 0, 'prepared': 0, 'active': 0}
    current, progress, detail_log = '', {}, ''
    for candidate in sorted((quantization / 'candidates').glob('*')):
        if not candidate.is_dir() or '.failed-' in candidate.name:
            continue
        state = jobs.read_optional(candidate / 'status.json')
        kind = state.get('status')
        if kind in counts:
            counts[kind] += 1
            if kind == 'prepared' and job['state'] in ('running', 'starting') and not current:
                preparation = service_preparation(candidate)
                if preparation:
                    current = candidate.name + ' · ' + preparation
                    progress = {'message': '服务启动尚无请求进度；详细加载与编译过程见 server.log'}
                    detail_log = str(candidate / 'server.log')
        elif kind:
            counts['active'] += 1
            current = candidate.name + ' · ' + {'measuring': '评测', 'quantizing_or_loading': '量化/加载'}.get(kind, kind)
            if kind == 'quantizing_or_loading':
                progress = jobs.read_optional(candidate / 'backend_progress.json')
                detail_log = str(candidate / 'quantization.log')
            else:
                detail_log = str(candidate / 'server.log')
                paths = list((candidate / 'datasets').glob('*/progress.json'))
                paths += list((candidate / 'datasets').glob('*/quality/progress.json'))
                paths += list((candidate / 'datasets').glob('*/performance/round-*/progress.json'))
                if paths:
                    newest = max(paths, key=lambda path: path.stat().st_mtime)
                    progress = jobs.read_optional(newest)
                    current += ' · ' + str(newest.parent.relative_to(candidate))
    inference = root / 'runs/inference' / (run_id + '-accel')
    if job['workflow'].get('status') == 'acceleration':
        active = jobs.read_optional(inference / 'active.json')
        candidate_progress = jobs.read_optional(inference / 'progress.json')
        phase = active.get('phase', '服务准备')
        current = {'attention': '注意力内核', 'kernels': '量化矩阵内核',
                   'graphs': '编译与CUDA Graph', 'budgets': '调度预算',
                   'mtp': 'MTP预测解码', 'final': '最终复测'}.get(phase, phase) + ' · ' + active.get('profile_id', '')
        progress = (candidate_progress if candidate_progress.get('phase') == active.get('phase')
                    and candidate_progress.get('profile_id') == active.get('profile_id') else {})
        detail_log = (str(inference / 'trials' / active['phase'] / active['profile_id'] / 'server.log')
                      if active.get('phase') and active.get('profile_id') else '')
    return {'job': job, 'counts': counts, 'current': current, 'progress': progress, 'detail_log': detail_log,
            'skip_quantization': skip_quantization, 'can_reselect': can_reselect}


def stage_path(data):
    # Keep the standalone read-only viewer independent of newer workspace modules.
    if data.get('skip_quantization'):
        return '环境检查 → 已有模型 → 加速对比 → 选择方案 → 部署'
    return '环境检查 → 量化与评测 → 选择模型 → 加速对比 → 部署'


def render(data, resources, run_id, width=100, height=30, colors=True, *, read_only=False):
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from .terminal import duration
    from .diagnostics import explain
    output = io.StringIO()
    console = Console(file=output, width=max(38, width), force_terminal=colors, color_system='standard' if colors else None)
    job, counts, progress = data['job'], data['counts'], data['progress']
    title = Text.assemble(('InferBoost', 'bold cyan'), ('  ' + run_id, 'dim'))
    table = Table.grid(padding=(0, 2))
    table.add_column(style='dim', no_wrap=True)
    table.add_column(overflow='fold')
    table.add_row('状态', Text(STATES.get(job['state'], job['state']), style='red' if job['state'] == 'failed' else 'cyan'))
    table.add_row('当前工作', Text(data['current'] or '准备 / 等待选择'))
    table.add_row('累计耗时', duration(time.time() - job.get('started_at', job.get('submitted_at', time.time()))))
    if progress.get('total'):
        done, total = progress.get('completed', 0), progress['total']
        filled = min(20, int(done / total * 20))
        table.add_row('本轮进度', Text('━' * filled + '─' * (20 - filled) + f'  {done}/{total}', style='cyan'))
    else:
        table.add_row('本轮状态', Text(progress.get('message', '总量未知时仅显示状态，不估算完成百分比')))
    if data.get('skip_quantization'):
        table.add_row('模型来源', '复用已提供模型，直接测试加速')
    else:
        table.add_row('量化候选', f"完成 {counts['completed']} · 准备好 {counts['prepared']} · 跳过 {counts['skipped']} · 失败 {counts['failed']}")
    if resources:
        table.add_row('GPU', ' / '.join(f"{g['index']}: {g['used_gib']:.1f}/{g['total_gib']:.0f} GiB" for g in resources['gpus']) or '未检测到')
        memory = resources['memory']
        table.add_row('容器内存', f"{memory['used_gib']:.1f} / {memory['limit_gib']:.1f} GiB" if memory['used_gib'] is not None else '占用未知')
        table.add_row('磁盘可用', f"{resources['storage']['free_gib']:.1f} GiB")
    sections = [title, Text(stage_path(data), style='dim'), table]
    if data.get('refresh_error'):
        sections.append(Text('状态刷新暂不可用：' + data['refresh_error'], style='yellow'))
    if job.get('error') or job['workflow'].get('error'):
        reason, recovery = explain(job.get('error') or job['workflow']['error'])
        sections.append(Text(reason + '\n' + recovery, style='yellow'))
    controls = ('L 日志  R 结果  Q 退出查看（只读）' if read_only
                else 'L 日志  R 结果  C 选择/续跑' + ('  B 重选模型' if data.get('can_reselect', True) else '')
                     + '  Q 退出查看  S 停止任务')
    sections.append(Text(controls + '\n退出查看不会停止计算；容器重建后需重新挂载工作卷并续跑。', style='dim'))
    console.print(Panel(Group(*sections), border_style='cyan', padding=(1, 1)))
    lines = output.getvalue().splitlines()
    if len(lines) > height - 2:
        return '\n'.join(lines[:max(1, height - 4)] + ['窗口较小，可放大终端；' + controls])
    return output.getvalue()


def tail(path, size=LOG_TAIL_BYTES):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            return stream.read().decode(errors='replace')
    except OSError as error:
        return str(error)


def log_source(data, detailed=False):
    if detailed:
        return '当前阶段详细日志', data.get('detail_log', '')
    return '任务主日志', data['job'].get('log', '')


def read_log(path):
    """Reopen a bounded tail each refresh so truncation and rotation are safe."""
    if not path:
        return {'path': '', 'text': '当前阶段暂无详细日志；Tab 可切回任务主日志。', 'updated_at': None}
    try:
        updated_at = Path(path).stat().st_mtime
    except OSError:
        updated_at = None
    text = tail(path)
    # Render untrusted output as plain text, including Rich markup and terminal
    # control sequences. Keep only the latest carriage-return progress update.
    text = re.sub(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)', '', text)
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    text = re.sub(r'\x1b[@-_]', '', text)
    text = text.replace('\r\n', '\n')
    text = '\n'.join(line.rsplit('\r', 1)[-1] for line in text.split('\n'))
    text = ''.join(character for character in text if character in '\n\t' or character.isprintable())
    return {'path': path, 'text': text or '日志已创建，等待写入。', 'updated_at': updated_at}


def render_log(data, log, run_id, detailed=False, width=100, height=30, colors=True):
    from rich.console import Console
    from rich.text import Text
    output = io.StringIO()
    console = Console(file=output, width=max(1, width), force_terminal=colors,
                      color_system='standard' if colors else None)
    label, path = log_source(data, detailed)
    console.print(Text(label + '：' + (path or '等待当前阶段'), style='cyan'), no_wrap=True, overflow='ellipsis')
    updated = log.get('updated_at')
    age = f'距最近写入 {max(0, int(time.time() - updated))} 秒' if updated is not None else '等待日志文件'
    rows = max(0, min(MAX_LOG_LINES, height - 2))
    console.print(Text(f'实时日志 · 最新最多 {rows} 行 / 每秒刷新 · ' + age, style='dim'), no_wrap=True, overflow='ellipsis')
    body = log.get('text', '正在读取日志…') if log.get('path') == path else '正在切换日志…'
    # One record occupies one screen row. Long records are truncated visually;
    # the full log stays on disk. Old lines disappear, never enter scrollback.
    if rows:
        for line in body.splitlines()[-rows:]:
            console.print(Text(line), no_wrap=True, overflow='ellipsis')
    return '\n'.join(output.getvalue().splitlines()[:max(0, height)])


def render_pinned(data, resources, run_id, width=100, height=9, colors=True):
    """Compact fixed header leaves room for log rows even in short terminals."""
    from rich.console import Console
    from rich.text import Text
    from .terminal import duration
    job, counts, progress = data['job'], data['counts'], data['progress']
    output = io.StringIO()
    console = Console(file=output, width=max(1, width), force_terminal=colors,
                      color_system='standard' if colors else None)
    elapsed = duration(time.time() - job.get('started_at', job.get('submitted_at', time.time())))
    lines = [Text('InferBoost · ' + run_id, style='bold cyan'),
             Text(stage_path(data), style='dim'),
             Text('状态：' + STATES.get(job['state'], job['state']) + '  ·  已用 ' + elapsed, style='cyan'),
             Text('当前：' + (data['current'] or '准备 / 等待选择'))]
    if progress.get('total'):
        lines.append(Text(f"进度：{progress.get('completed', 0)} / {progress['total']}", style='cyan'))
    else:
        lines.append(Text('进度：' + progress.get('message', '等待阶段进度；未知总量不估算百分比')))
    if data.get('skip_quantization'):
        lines.append(Text('模型：复用已提供模型，直接测试加速'))
    else:
        lines.append(Text(f"候选：完成 {counts['completed']} · 准备好 {counts['prepared']} · 跳过 {counts['skipped']} · 失败 {counts['failed']}"))
    if resources:
        gpu = ' / '.join(f"{g['index']}: {g['used_gib']:.1f}/{g['total_gib']:.0f}GiB" for g in resources['gpus']) or '未检测到'
        memory = resources['memory']
        used = memory.get('used_gib')
        limit = memory.get('limit_gib')
        memory_text = f'{used:.1f}/{limit:.1f}GiB' if used is not None and limit is not None else '未知'
        lines.append(Text(f"GPU {gpu} · 内存 {memory_text} · 磁盘剩余 {resources['storage']['free_gib']:.1f}GiB", style='dim'))
    error = data.get('refresh_error') or job.get('error') or job['workflow'].get('error')
    if error:
        lines.append(Text(str(error).replace('\n', ' '), style='yellow'))
    for line in lines[:max(0, height - 1)]:
        console.print(line, no_wrap=True, overflow='ellipsis')
    console.print(Text('─' * max(1, width), style='cyan'), no_wrap=True)
    return output.getvalue().rstrip('\n')


def watch(root, run_id, *, read_only=False):
    import os
    import sys
    from prompt_toolkit import Application
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from . import ui
    if not sys.stdin.isatty():
        print(render(snapshot(root, run_id), {}, run_id, colors=False, read_only=read_only))
        return
    while True:
        data = snapshot(root, run_id)
        resources = {}
        showing_log, detailed_log = read_only, False
        log = {}
        resource_error = ''
        keys = KeyBindings()
        for key in ['q', 'r', 'c-c', 'c-d'] + ([] if read_only else ['s', 'c'] + (['b'] if data.get('can_reselect', True) else [])):
            def action(event, selected=key):
                event.app.exit(result='q' if selected in ('c-c', 'c-d') else selected)
            keys.add(key)(action)
        @keys.add('l')
        def toggle_log(event):
            nonlocal showing_log
            showing_log = not showing_log
            event.app.invalidate()
        @keys.add('enter')
        def close_log(event):
            nonlocal showing_log
            showing_log = False
            event.app.invalidate()
        @keys.add('tab')
        def switch_log(event):
            nonlocal detailed_log
            if showing_log:
                detailed_log = not detailed_log
                event.app.invalidate()
        def header_height():
            rows = app.output.get_size().rows
            return min(9, max(3, rows // 3)) if showing_log else max(1, rows - 1)
        def header_content():
            size = app.output.get_size()
            current_data = dict(data, refresh_error=data.get('refresh_error') or resource_error)
            if showing_log:
                return ANSI(render_pinned(current_data, resources, run_id, size.columns,
                                          header_height(), 'NO_COLOR' not in os.environ))
            return ANSI(render(current_data, resources, run_id, size.columns, size.rows,
                               'NO_COLOR' not in os.environ, read_only=read_only))
        def log_content():
            size = app.output.get_size()
            height = max(0, size.rows - header_height() - 1)
            return ANSI(render_log(data, log, run_id, detailed_log, size.columns, height,
                                   'NO_COLOR' not in os.environ))
        def footer_content():
            navigation = ('L / Enter 收起日志 · Tab 切换日志' if showing_log else 'L 展开实时日志')
            return navigation + ' · R 结果 · Q 退出查看' + ('（只读）' if read_only else
                ' · C 选择/续跑' + (' · B 重选模型' if data.get('can_reselect', True) else '') + ' · S 停止任务')
        header_window = Window(FormattedTextControl(header_content), height=header_height,
                               always_hide_cursor=True, wrap_lines=False)
        log_window = Window(FormattedTextControl(log_content), always_hide_cursor=True, wrap_lines=False)
        footer_window = Window(FormattedTextControl(footer_content), height=1,
                               always_hide_cursor=True, wrap_lines=False)
        app = Application(layout=Layout(HSplit([
            header_window, ConditionalContainer(log_window, filter=Condition(lambda: showing_log)), footer_window])),
            key_bindings=keys, full_screen=True)
        async def refresh():
            nonlocal data, log
            while True:
                try:
                    data = await asyncio.to_thread(snapshot, root, run_id)
                    if showing_log:
                        log = await asyncio.to_thread(read_log, log_source(data, detailed_log)[1])
                except (OSError, ValueError) as error:
                    data = dict(data, refresh_error=str(error))
                app.invalidate()
                await asyncio.sleep(REFRESH_SECONDS)
        async def refresh_resources():
            nonlocal resources, resource_error
            while True:
                try:
                    resources = await asyncio.to_thread(preflight.inspect_resources, root)
                    resource_error = ''
                except (OSError, ValueError) as error:
                    resource_error = '资源采样暂不可用：' + str(error)
                app.invalidate()
                await asyncio.sleep(RESOURCE_REFRESH_SECONDS)
        def start_refresh():
            app.create_background_task(refresh())
            app.create_background_task(refresh_resources())
        result = app.run(pre_run=start_refresh)
        if result == 'q':
            ui.console.print('已退出查看。后台任务状态未改变；可从任务中心重新进入。', style='dim')
            return
        if result == 'r':
            if not data.get('skip_quantization') or data.get('can_reselect', True):
                ui.show_report(root, run_id)
            ui.show_report(root, run_id, 'inference')
            try:
                ui.text_input('Enter 回看板', optional=True)
            except (ui.Back, ui.Cancel):
                pass
        elif result == 's':
            try:
                if ui.choose('停止后保留已完成候选，当前候选可能需要重做',
                             [('back', '继续运行，回到看板', None), ('stop', '停止这个任务并清理其服务', None)]) == 'stop':
                    jobs.request_stop(root, run_id)
            except (ui.Back, ui.Cancel):
                pass
            except (ValueError, OSError) as error:
                ui.console.print(str(error), markup=False, style='yellow')
                try:
                    ui.text_input('Enter 回看板', optional=True)
                except (ui.Back, ui.Cancel):
                    pass
        elif result in ('c', 'b'):
            try:
                current = jobs.status(root, run_id)
                if current['alive']:
                    ui.console.print('任务仍在运行，请等待当前阶段完成。', style='yellow')
                    continue
                state = current['workflow'].get('status')
                if state == 'completed' and not data.get('can_reselect', True):
                    ui.console.print('加速与部署验收已完成；部署包位于 runs/workflows/' + run_id + '/deployment')
                    ui.text_input('Enter 回看板', optional=True)
                    continue
                if result == 'b' or state == 'completed':
                    branch = choose_retest(root, run_id)
                    if branch:
                        run_id = branch
                    continue
                candidate = profile = None
                if state in ('awaiting_model_selection', 'awaiting_profile_selection'):
                    stage = 'quantization' if state == 'awaiting_model_selection' else 'inference'
                    options = ui.show_report(root, run_id, stage)
                    selected = ui.choose('选择后继续下一阶段', options)
                    if stage == 'quantization':
                        candidate = selected
                    else:
                        profile = selected
                elif ui.choose('续跑将保留已完成候选；不保证量化层级续算',
                               [('resume', '按原配置继续', None), ('back', '返回看板', None)]) != 'resume':
                    continue
                request_path = Path(root) / 'runs/workflows' / run_id / 'request.json'
                arguments = (jobs.continuation(root, run_id, candidate, profile) if request_path.exists()
                             else jobs.read_optional(Path(root) / 'jobs' / run_id / 'request.json')['arguments'])
                jobs.submit(root, arguments)
            except (ui.Back, ui.Cancel):
                pass
            except (ValueError, OSError, RuntimeError, KeyError) as error:
                ui.console.print(str(error), markup=False, style='red')
                try:
                    ui.text_input('Enter 回看板', optional=True)
                except (ui.Back, ui.Cancel):
                    pass
