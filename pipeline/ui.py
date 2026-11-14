"""Rich rendering and prompt_toolkit controls, with one input owner at a time."""
from pathlib import Path
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from prompt_toolkit import Application, PromptSession
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

console = Console(highlight=False)
STYLE = Style.from_dict({'accent': '#54c8d7 bold', 'muted': 'ansibrightblack',
                        'disabled': 'ansibrightblack', 'error': 'ansired', 'selected': 'reverse'})


class Back(Exception):
    pass


class Cancel(Exception):
    pass


def heading(title, subtitle=''):
    console.print(Panel(Text.assemble((title, 'bold cyan'), ('\n' + subtitle, 'dim') if subtitle else ''),
                        border_style='cyan', padding=(0, 1)))


def text_input(label, default='', path=False, optional=False):
    session = PromptSession(completer=PathCompleter(expanduser=True) if path else None,
                            complete_while_typing=False)
    console.print(Text('Tab 补全路径 · :b 返回 · :q 返回菜单' + (' · :clear 清空' if optional else ''), style='dim'))
    answer = session.prompt(label + ' › ', default=str(default)).strip()
    if answer == ':b':
        raise Back()
    if answer == ':q':
        raise Cancel()
    if answer == ':clear':
        if not optional:
            raise ValueError('此项不能为空')
        return ''
    if not answer and not optional:
        raise ValueError('此项不能为空')
    return answer


def choose(title, options, default=None, multiple=False):
    """options: (value, label, disabled_reason). Printable keys filter the list."""
    if not options:
        raise ValueError('没有可选项')
    selected = set(default or []) if multiple else set()
    position = next((i for i, row in enumerate(options) if row[0] == default), 0) if not multiple else 0
    query = ''
    message = ''
    keys = KeyBindings()
    def visible():
        return [row for row in options if query.casefold() in (str(row[0]) + row[1]).casefold()]
    def content():
        rows = visible()
        fragments = [('class:accent', title + '\n'), ('class:muted', '搜索：' + query + '\n')]
        # Scroll around selection without clearing terminal history.
        start = max(0, position - 7)
        for index, (value, label, disabled) in enumerate(rows[start:start + 12], start):
            pointer = '› ' if index == position else '  '
            marker = ('[x] ' if value in selected else '[ ] ') if multiple else ''
            role = 'class:disabled' if disabled else 'class:accent' if index == position else ''
            fragments.append((role, pointer + marker + label + (' · ' + disabled if disabled else '') + '\n'))
        if not rows:
            fragments.append(('class:muted', '没有匹配项，按退格修改搜索\n'))
        hint = '↑↓ 移动 · ' + ('Space 勾选 · ' if multiple else '') + 'Enter 确认 · Esc 返回 · Ctrl+C 回菜单'
        fragments.extend([('class:muted', hint + '\n'), ('class:error', message)])
        return FormattedText(fragments)
    @keys.add('up')
    def up(event):
        nonlocal position, message
        position = max(0, position - 1)
        message = ''
    @keys.add('down')
    def down(event):
        nonlocal position, message
        position = min(max(0, len(visible()) - 1), position + 1)
        message = ''
    @keys.add('escape')
    def back(event):
        event.app.exit(exception=Back())
    @keys.add('c-c')
    def cancel(event):
        event.app.exit(exception=Cancel())
    @keys.add('c-d')
    def eof(event):
        event.app.exit(exception=Cancel())
    @keys.add('backspace')
    def erase(event):
        nonlocal query, position
        query, position = query[:-1], 0
    @keys.add(' ')
    def toggle(event):
        nonlocal message, query, position
        if not multiple:
            query += ' '
            position = 0
            return
        rows = visible()
        if rows:
            value, _, disabled = rows[position]
            if disabled:
                message = disabled
            else:
                selected.remove(value) if value in selected else selected.add(value)
                message = ''
    @keys.add('enter')
    def accept(event):
        nonlocal message
        if multiple:
            allowed = {row[0] for row in options if not row[2]}
            chosen = [row[0] for row in options if row[0] in selected & allowed]
            if chosen:
                event.app.exit(result=chosen)
            else:
                message = '至少选择一个兼容项'
        elif visible():
            row = visible()[position]
            if row[2]:
                message = row[2]
            else:
                event.app.exit(result=row[0])
    @keys.add('<any>')
    def search(event):
        nonlocal query, position, message
        if event.data.isprintable():
            query += event.data
            position, message = 0, ''
    control = FormattedTextControl(content)
    application = Application(layout=Layout(HSplit([Window(control, always_hide_cursor=True)])),
                              key_bindings=keys, style=STYLE, full_screen=False,
                              mouse_support=False)
    return application.run()


def key_values(title, values):
    table = Table.grid(padding=(0, 2))
    table.add_column(style='dim', no_wrap=True)
    table.add_column(overflow='fold')
    for key, value in values:
        table.add_row(Text(str(key)), Text(str(value)))
    console.print(Panel(table, title=title, border_style='cyan'))


def show_report(root, run_id, stage='quantization'):
    from .common import read_json
    directory = Path(root) / ('runs' if stage == 'quantization' else 'runs/inference') / (run_id + ('-quant' if stage == 'quantization' else '-accel'))
    if stage == 'quantization' and not directory.exists():
        request = Path(root) / 'runs/workflows' / run_id / 'request.json'
        lineage = read_json(request).get('retest_source') if request.is_file() else None
        if lineage:
            directory = Path(lineage['quantization_directory'])
    path = directory / 'report.json'
    if not path.is_file():
        console.print('当前阶段尚未生成报告。', style='yellow')
        return []
    report = read_json(path)
    table = Table(title='量化结果 · 本测试集得分' if stage == 'quantization' else '加速结果 · 已完成复测',
                  header_style='bold cyan', box=__import__('rich.box', fromlist=['SIMPLE']).SIMPLE)
    options = []
    if stage == 'quantization':
        for column in ['候选 / 精度', '数据', '得分%', '权重GiB', '采样显存峰值GiB', 'TTFT秒', 'E2E秒', 'TPS']:
            table.add_column(column)
        for row in report['overview']:
            def number(key):
                if row.get(key) is None:
                    return '—'
                value = f'{row[key]:.3f}'
                if row.get(key + '_std') is not None:
                    value += f" ± {row[key + '_std']:.3f}"
                return value
            table.add_row(Text(row['candidate'] + ' / ' + row['quantization_label']), Text(row['dataset']),
                          *[number(k) for k in ['score_pct', 'weight_gib', 'service_gpu_sample_peak_gib', 'ttft_s', 'e2e_s', 'tps']])
            if report['selectable'].get(row['candidate']) and row['candidate'] not in [o[0] for o in options]:
                gate = report['candidate_gates'][row['candidate']]['passed']
                options.append((row['candidate'], row['quantization_label'] + (' · 达到本次质量门槛' if gate else ' · 有质量退化或未评分'), None))
        recommendation = report['recommendation']
        reason = recommendation['basis'] + '；偏好=' + {'size': '权重大小', 'latency': '延迟', 'quality': '本测试集得分'}.get(recommendation['objective'], recommendation['objective'])
        selected = recommendation['candidate']
    else:
        for column in ['配置', '缓存', '本集得分(0–1)', 'TTFT秒', 'E2E秒', 'TPS']:
            table.add_column(column)
        for row in report['rows']:
            table.add_row(Text(row['label']), row['cache'],
                          *['—' if row.get(k) is None else f'{row[k]:.3f}'
                            for k in ['quality_score', 'ttft_s_mean', 'e2e_s_mean', 'decode_tps_mean']])
            if report['quality_gates'].get(row['profile']) and row['profile'] not in [o[0] for o in options]:
                options.append((row['profile'], row['label'], None))
        selected, reason = report.get('recommendation'), report['objective']
    console.print(table)
    if report.get('measurement_note'):
        console.print(Text(report['measurement_note'], style='dim'))
    if report.get('experience', {}).get('smoke_only'):
        console.print('快速体验：仅验证链路，不构成正式部署推荐。', style='yellow')
    console.print(Text('建议：' + str(selected or '暂无') + '。' + reason))
    console.print(Text('报告：' + str(directory / 'report.html') + '\nMarkdown：' + str(directory / 'report.md'), style='dim'))
    return options
