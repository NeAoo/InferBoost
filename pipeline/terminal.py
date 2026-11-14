"""Small, dependency-free terminal UI; counters describe work, never guessed time."""
import os
import re
import shutil
import sys
import threading
import time
import unicodedata

from .product import NAME, CHINESE_NAME, VERSION


COLORS = {'accent': '36', 'success': '32', 'warning': '33', 'error': '31', 'muted': '90'}
OUTPUT_LOCK = threading.RLock()
LOGO = r'''
  ___        __           ____                  _
 |_ _|_ __  / _| ___ _ __ | __ )  ___   ___  ___| |_
  | || '_ \| |_ / _ \ '__||  _ \ / _ \ / _ \/ __| __|
  | || | | |  _|  __/ |   | |_) | (_) | (_) \__ \ |_
 |___|_| |_|_|  \___|_|   |____/ \___/ \___/|___/\__|
'''.strip('\n')


def interactive(stream=None):
    stream = sys.stdout if stream is None else stream
    return bool(getattr(stream, 'isatty', lambda: False)()) and os.environ.get('TERM') != 'dumb'


def clean(value):
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', str(value))
    return ''.join(character for character in text if character >= ' ' and character != '\x7f')


def style(value, role='accent', bold=False, stream=None):
    text = clean(value)
    if not interactive(stream) or 'NO_COLOR' in os.environ:
        return text
    code = ('1;' if bold else '') + COLORS.get(role, '0')
    return f'\033[{code}m{text}\033[0m'


def width_limit(text, width):
    result, cells = [], 0
    for character in clean(text):
        size = 0 if unicodedata.combining(character) else 2 if unicodedata.east_asian_width(character) in 'WF' else 1
        if cells + size > max(0, width - 1):
            return ''.join(result) + '…'
        result.append(character)
        cells += size
    return ''.join(result)


def message(text, role='accent', bold=False):
    with OUTPUT_LOCK:
        print(style(text, role, bold), flush=True)


def section(title, detail=None):
    print()
    message('── ' + title + ' ──', bold=True)
    if detail:
        message(detail, 'muted')


def brand(root=None):
    columns = shutil.get_terminal_size((80, 24)).columns
    print()
    if interactive() and columns >= 60:
        for line in LOGO.splitlines():
            message(line, bold=True)
    message(f'  {NAME} · {CHINESE_NAME}  v{VERSION}', bold=True)
    message('  量化  →  评测与选择  →  推理加速  →  部署', 'muted')
    if root is not None:
        message('工作目录：' + str(root))
        usage = shutil.disk_usage(root)
        message(f'可用空间：{usage.free / 1024**3:.1f} GiB；跨实例保留需要挂载同一持久卷。',
                'warning' if usage.free < 30 * 1024**3 else 'muted')
    print()


def step(number, total, title, detail=None):
    section(f'配置 {number}/{total} · {title}', detail)
    message('Enter 保留当前值  ·  :b 返回上一步  ·  :q 返回主菜单  ·  :clear 清空可选项', 'muted')


def flow_path(skip_quantization=False):
    if skip_quantization:
        return '环境检查 → 已有模型 → 加速对比 → 选择方案 → 部署'
    return '环境检查 → 量化与评测 → 选择模型 → 加速对比 → 部署'


def duration(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}'


class Progress:
    """One measured scope. Concurrent jobs use live=False to keep logs readable."""
    def __init__(self, label, total=None, unit='项', live=True, interval=10, stream=None):
        self.label, self.total, self.unit = label, total, unit
        self.stream = sys.stdout if stream is None else stream
        self.live = live and interactive(self.stream)
        self.interval = interval
        self.completed, self.detail = 0, ''
        self.started = time.monotonic()
        self.last_draw = None
        self.closed = False

    def update(self, completed=None, detail=None, force=False):
        if completed is not None:
            if completed < 0 or (self.total is not None and completed > self.total):
                raise ValueError('进度计数超出当前任务范围')
            self.completed = completed
        if detail is not None:
            self.detail = detail
        now = time.monotonic()
        if force or self.last_draw is None or now - self.last_draw >= (0.2 if self.live else self.interval):
            self._draw(now)

    def _draw(self, now, outcome=None):
        elapsed = duration(now - self.started)
        if self.total:
            ratio = self.completed / self.total
            filled = int(ratio * 16)
            indicator = '[' + '#' * filled + '-' * (16 - filled) + ']'
            counter = f'{self.completed}/{self.total} {self.unit} ({ratio:.0%})'
        else:
            indicator, counter = '|/-\\'[int((now - self.started) * 2) % 4], '处理中'
        line = f'{self.label}  {indicator} {counter}  已用 {elapsed}'
        if outcome:
            line += ' · ' + outcome
        elif self.detail:
            line += ' · ' + self.detail
        role = 'error' if outcome in ('失败', '已中断') else 'success' if outcome == '完成' else 'accent'
        with OUTPUT_LOCK:
            if self.live:
                line = width_limit(line, shutil.get_terminal_size((80, 24)).columns - 1)
                self.stream.write('\r\033[2K' + style(line, role, stream=self.stream) + ('\n' if outcome else ''))
            else:
                self.stream.write(style(line, role, stream=self.stream) + '\n')
            self.stream.flush()
        self.last_draw = now

    def close(self, outcome=None):
        if not self.closed:
            outcome = outcome or ('完成' if self.total is None or self.completed == self.total else '阶段结束')
            self._draw(time.monotonic(), outcome)
            self.closed = True

    def __enter__(self):
        self.update(force=True)
        return self

    def __exit__(self, kind, error, traceback):
        self.close('已中断' if kind and issubclass(kind, (KeyboardInterrupt, SystemExit)) else '失败' if kind else None)


if __name__ == '__main__':
    brand()
    message('输入 quickstart 打开向导；首次使用请设置 PIPELINE_WORKSPACE 为持久卷中的新目录。')
