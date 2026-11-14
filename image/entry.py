"""Select, validate and enter an isolated writable image workspace."""
import json
import os
from pathlib import Path
import shlex
import sys

SOURCE = Path('/opt/pipeline')
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def matches_source(source, destination):
    from image.workspace import MANIFEST
    try:
        return json.loads((destination / MANIFEST).read_text()) == json.loads((source / MANIFEST).read_text())
    except (OSError, ValueError):
        return False


def choose_workspace(source, destination, preference_file, current=None, plain=False):
    """Return a confirmed initialized directory, or None; never touch old jobs."""
    from image.workspace import initialize, recent_workspaces, remember, validate_destination
    from pipeline.preflight import storage_snapshot, workspace_choices
    ui = None
    if not plain:
        from pipeline import ui
    def display(message, error=False):
        if ui:
            ui.console.print(message, style='red' if error else None, markup=False)
        else:
            print(message)
    def ask_path(default):
        if ui:
            return ui.text_input('工作目录', str(default), path=True)
        answer = input(f'工作目录 [{default}]（:b / :q 返回）：').strip()
        return answer or str(default)
    def choose(title, options):
        if ui:
            return ui.choose(title, [(value, label, None) for value, label in options])
        print(title)
        for index, (_, label) in enumerate(options, 1):
            print(f'  {index}  {label}')
        answer = input('选择（:b / :q 返回）：').strip()
        if answer in (':b', ':q', '0'):
            return 'cancel'
        if not answer.isdigit() or not 1 <= int(answer) <= len(options):
            raise ValueError('请输入列表中的编号。')
        return options[int(answer) - 1][0]

    if ui:
        ui.heading('工作目录 · 更换存储路径', '新任务写入所选目录；原任务、权重、报告和后台进程保持原位置。')
    else:
        display('工作目录 · 更换存储路径\n新任务写入所选目录；原任务、权重、报告和后台进程保持原位置。')
    display('当前目录：' + str(current or destination))
    candidates = workspace_choices()
    if candidates:
        display('可访问目录：' + '、'.join(map(str, candidates)) + '（是否持久化以平台挂载配置为准）')
    display('支持新空目录或本镜像同版本的工作目录；不会自动迁移模型、缓存或报告。')
    cancellation = (KeyboardInterrupt, EOFError) + ((ui.Back, ui.Cancel) if ui else ())
    while True:
        try:
            recent = recent_workspaces(source, preference_file)
            options = [('path', '输入新路径 / 已有目录')]
            options += [(str(index), '最近使用 · ' + path) for index, path in enumerate(recent)]
            options.append(('cancel', '返回，不更改'))
            action = choose('选择存储位置', options)
            if action == 'cancel':
                return None
            value = ask_path(destination) if action == 'path' else recent[int(action)]
            if value in (':b', ':q'):
                return None
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if not value.strip():
                raise ValueError('路径不能为空。')
            destination = validate_destination(source, Path(value).expanduser().resolve(), current)
            info = storage_snapshot(destination)
            values = [('目标目录', destination), ('磁盘可用', f"{info['free_gib']:.1f} GiB"),
                      ('挂载点', info['mount'] or '未知'), ('持久性', info['persistence'])]
            if ui:
                ui.key_values('切换前检查', values)
            else:
                for key, item in values:
                    display(f'{key}：{item}')
            display('旧目录的历史任务仍在旧目录；切回旧目录即可继续查看。切换不会停止后台任务。')
            action = choose('确认工作目录', [('use', '使用此目录并返回主菜单'),
                                           ('change', '重新选择'), ('cancel', '返回，不更改')])
            if action == 'cancel':
                return None
            if action == 'change':
                continue
            initialize(source, destination, current)
            # Probe existing workspaces too; a valid manifest does not imply write access.
            storage_snapshot(destination, check_write=True)
            remember(source, destination, preference_file)
            display('已选择工作目录：' + str(destination))
            display('已记住选择（保存在当前用户主目录）。显式 PIPELINE_WORKSPACE 优先；\n'
                    '如原终端导出过旧路径，请改为以下命令；新容器也可用它进入同一目录：\n  '
                    'export PIPELINE_WORKSPACE=' + shlex.quote(str(destination)) + '\n  quickstart')
            return destination
        except cancellation:
            return None
        except (OSError, ValueError) as error:
            display(str(error), error=True)


def launch(destination, arguments):
    os.environ['PIPELINE_WORKSPACE'] = str(destination)
    os.environ['PIPELINE_PYTHON'] = sys.executable
    print('工作目录：' + str(destination), flush=True)
    program = str(destination / 'quickstart')
    os.execv(program, [program, *arguments])


def enter(source=SOURCE, arguments=None):
    from image.workspace import initialize, PREFERENCE, resolve_destination
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    source = Path(source)
    preference_file = Path.home() / PREFERENCE
    destination = resolve_destination(source, preference_file, os.environ)
    manage = bool(arguments and arguments[0] == 'workspace')
    if manage:
        arguments.pop(0)
        if arguments not in ([], ['--plain']):
            raise ValueError('用法：quickstart workspace [--plain]')
    interactive = manage or not arguments or arguments[0] in ('menu', '--plain')
    if manage and not sys.stdin.isatty():
        raise ValueError('更换目录需要交互终端，请使用 docker exec -it 或显式设置 PIPELINE_WORKSPACE。')
    plain = '--plain' in arguments or os.environ.get('TERM') == 'dumb'
    if interactive and sys.stdin.isatty() and (manage or not matches_source(source, destination)):
        destination = choose_workspace(source, destination, preference_file, plain=plain)
        if destination is None:
            return 0
    else:
        initialize(source, destination)
    launch(destination, arguments)


if __name__ == '__main__':
    try:
        raise SystemExit(enter())
    except (ValueError, OSError) as error:
        raise SystemExit('WORKSPACE_ERROR: ' + str(error) + '\n可运行 quickstart workspace 重新选择；旧结果保留。')
