"""Dependency-free terminal menu; all real work uses the existing workflow."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

from .common import identifier, read_json
from .inference.defaults import defaults_text
from .product import NAME, CHINESE_NAME, VERSION
from .quantization_labels import describe_quantization
from . import terminal
from .workflow import RECIPE_NAMES, choose_recipes, resolve_config


MENU = '''
  1  新模型：量化 → 评测 → 选择 → 加速 → 部署包
  2  已量化模型：直接进行加速测试与部署
  3  查看历史任务、报告和量化模型位置
  4  继续已有完整流程任务
  5  启动已选定的部署包
  6  环境自检与存储说明
  7  工作目录 · 更换存储路径
  0  退出
'''


def banner(root):
    terminal.brand(root)


def menu():
    terminal.section('选择工作流')
    for line in MENU.strip().splitlines():
        number, title = line.strip().split('  ', 1)
        print('  ' + terminal.style(number, bold=True) + '  ' + title)
    print()


def ask(prompt, default=None):
    suffix = f' [{default}]' if default is not None else ''
    answer = input(prompt + suffix + '：').strip()
    return answer or default or ''


def path_input(value):
    # Terminal users often paste a quoted path containing spaces.
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        value = value[1:-1]
    if not value:
        raise ValueError('路径不能为空')
    return Path(value).expanduser().resolve()


def model_input(value, skip_quantization):
    path = path_input(value)
    if path.is_file():
        receipt = read_json(path)
        if not skip_quantization or receipt.get('stage') != 'quantization_selection':
            raise ValueError('完整流程需要模型目录；第二步也可传入 selected_model.json')
        if not (Path(receipt['model']) / 'config.json').is_file():
            raise ValueError('选择清单中的模型路径不可访问，请挂载原工作目录或传入模型目录')
        return path
    if not (path / 'config.json').is_file():
        raise ValueError('模型目录中找不到 config.json：' + str(path))
    config = read_json(path / 'config.json')
    if not skip_quantization and config.get('quantization_config'):
        raise ValueError('这个模型已有量化配置，请选菜单2，避免把量化模型再次量化')
    return path


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise ValueError('必须是正整数')
    return number


def normalize_gpus(value):
    devices = [int(item.strip()) for item in value.split(',')]
    if not devices or min(devices) < 0 or len(set(devices)) != len(devices):
        raise ValueError('GPU编号必须非负且不重复，例如0或0,1')
    return ','.join(map(str, devices))


class WizardCancelled(Exception):
    """Return to the menu without creating a task."""


class WorkflowWizard:
    def __init__(self, root, skip_quantization):
        self.root = Path(root)
        self.skip_quantization = skip_quantization
        self.values = {}
        self.index = 0
        default_length = read_json(resolve_config(root, None, 'inference'))['server']['max_model_len']
        self.questions = [
            ('model', '模型', '模型目录' + (' 或 selected_model.json' if skip_quantization else '（完整精度）'), '', False),
            ('mtp', '模型', 'MTP草稿目录；无则回车，与模型同目录输入 same', '', True),
        ]
        if not skip_quantization:
            self.questions.append(('recipes', '量化', '配方，逗号分隔；首次可选 awq4', 'all', False))
        self.questions += [
            ('dataset', '评测', '自定义测试JSON/JSONL；回车使用性能public100与质量quality100' if skip_quantization
             else '自定义测试JSON/JSONL；回车使用默认数据', '', True),
            ('gpus', '资源', '推理GPU编号（单卡）' if skip_quantization else 'GPU编号；多卡分担量化，性能统一在第一张卡测量', '0', False),
            ('max_jobs', '资源', '编译并发；不等于总内存限制', '2', False),
            ('max_model_len', '资源', '模型上下文上限', str(default_length), False),
            ('run_id', '任务', '本次任务名称', 'forge-' + time.strftime('%Y%m%d-%H%M%S'), False),
            ('auto_select', '选择方式', '加速后按推荐导出并验证部署包？y/n；n会等待选择加速方案' if skip_quantization
             else '按推荐自动选择并走完全流程？y/n；n会在选择点等待', 'n', False),
        ]

    def validate(self, key, value):
        if key == 'model':
            return str(model_input(value, self.skip_quantization))
        if key == 'mtp':
            if not value:
                return ''
            model = Path(self.values['model'])
            if value == 'same' and model.is_file():
                raise ValueError('传入选择清单时，请填写真实MTP目录；不能用same指代JSON文件')
            draft = model if value == 'same' else path_input(value)
            if not (draft / 'config.json').is_file():
                raise ValueError('MTP目录中找不到 config.json')
            return value if value == 'same' else str(draft)
        if key == 'recipes':
            choose_recipes(value)
        elif key == 'dataset' and value:
            from .quality import load_dataset
            path = path_input(value)
            load_dataset(path)
            return str(path)
        elif key == 'gpus':
            devices = normalize_gpus(value)
            if self.skip_quantization and ',' in devices:
                raise ValueError('直接加速当前使用单卡推理，请填写一个GPU编号')
            return devices
        elif key in ('max_jobs', 'max_model_len'):
            return str(positive_integer(value))
        elif key == 'run_id':
            value = identifier(value)
            if any(path.exists() for path in (self.root / 'runs/workflows' / value,
                    self.root / 'runs' / (value + '-quant'), self.root / 'runs/inference' / (value + '-accel'))):
                raise ValueError('任务名称已存在；请使用菜单4继续，或换一个新名称')
        elif key == 'auto_select':
            value = value.lower()
            if value not in ('y', 'n'):
                raise ValueError('请输入 y 或 n')
        return value

    def collect(self):
        if not self.values:
            terminal.section('开始前 · 默认加速设置')
            print(defaults_text())
            print('每步输出终端摘要，并保存 Markdown / HTML；默认由你选择加速方案。' if self.skip_quantization
                  else '每步输出终端摘要，并保存 Markdown / HTML；默认由你选择模型和加速方案。')
        while self.index < len(self.questions):
            key, group, prompt, default, optional = self.questions[self.index]
            terminal.step(self.index + 1, len(self.questions), group, terminal.flow_path(self.skip_quantization))
            if key == 'recipes':
                print('A16为默认BF16激活；A8/AFP8另外量化激活。')
                for name, recipe in RECIPE_NAMES.items():
                    print('  ' + terminal.style(name, bold=True) + ' = ' + describe_quantization(recipe)['quantization_label'])
                print('all尝试全部兼容配方；不支持的项目会明确跳过。')
            current = self.values.get(key, default)
            answer = input(terminal.style(prompt, bold=True) + (f' [{current}]' if current else '') + '：').strip()
            if answer == ':q':
                raise WizardCancelled()
            if answer == ':b':
                if self.index:
                    self.index -= 1
                else:
                    terminal.message('已经是第一步；输入 :q 可返回主菜单。', 'warning')
                continue
            if answer == ':clear':
                if not optional:
                    terminal.message('这一项必须填写，不能清空。', 'warning')
                    continue
                answer = ''
            else:
                answer = answer or current
            try:
                self.values[key] = self.validate(key, answer)
            except (ValueError, OSError, KeyError) as error:
                terminal.message(str(error) + '；请修改当前项，之前的输入已保留。', 'error')
                continue
            self.index += 1
        return self.arguments()

    def arguments(self):
        # Revalidate every field after back-navigation changes an earlier dependency.
        for index, (key, *_rest) in enumerate(self.questions):
            try:
                self.values[key] = self.validate(key, self.values[key])
            except (ValueError, OSError, KeyError) as error:
                terminal.message(str(error), 'error')
                self.index = index
                return self.collect()
        values = self.values
        arguments = ['--model', values['model']]
        if self.skip_quantization:
            arguments.append('--skip-quantization')
        if values['mtp']:
            arguments += ['--mtp-model', values['model'] if values['mtp'] == 'same' else values['mtp']]
        if not self.skip_quantization:
            arguments += ['--recipes', ','.join(item['recipe'].removeprefix('compressor_')
                                               for item in choose_recipes(values['recipes']))]
        if values['dataset']:
            arguments += ['--dataset', values['dataset']]
        for key in ('gpus', 'max_jobs', 'max_model_len', 'run_id'):
            arguments += ['--' + key.replace('_', '-'), values[key]]
        if values['auto_select'] == 'y':
            arguments.append('--auto-select')
        terminal.section('运行前确认', '所有参数确认后才创建任务。')
        labels = {'model': '模型', 'mtp': 'MTP', 'recipes': '量化配方', 'dataset': '测试集',
                  'gpus': 'GPU', 'max_jobs': '编译并发', 'max_model_len': '上下文上限',
                  'run_id': '任务名称', 'auto_select': '自动选择'}
        for key, *_rest in self.questions:
            print('  ' + terminal.style(labels[key], bold=True) + '：' + (values[key] or ('默认100条' if key == 'dataset' else '未启用')))
        print('  工作目录：' + str(self.root))
        run_id = values['run_id']
        if not self.skip_quantization:
            print('  量化权重：' + str(self.root / 'runs' / (run_id + '-quant') / 'candidates/<方案>/quantized'))
        print('  加速报告：' + str(self.root / 'runs/inference' / (run_id + '-accel')))
        print('  部署包：' + str(self.root / 'runs/workflows' / run_id / 'deployment'))
        return arguments


def workflow_arguments(root, skip_quantization):
    return WorkflowWizard(root, skip_quantization).collect()


def run_workflow(root, arguments, allow_back=False):
    print('\n等价命令：\n' + shlex.join([str(root / 'quickstart'), 'workflow', *arguments]))
    prompt = '开始运行？y/n' + ('；:b 返回修改，:q 返回主菜单' if allow_back else '')
    answer = ask(prompt, 'y').lower()
    if allow_back and answer == ':b':
        return 'back'
    if answer != 'y':
        print('已返回菜单，未启动任务。')
        return 'cancelled'
    from .workflow import main as workflow_main
    workflow_main(['--root', str(root), *arguments], install_signals=False)
    return 'started'


def history(root):
    entries = []
    for base in (root / 'runs', root / 'runs/inference', root / 'runs/workflows'):
        if not base.is_dir():
            continue
        for directory in base.iterdir():
            if directory.is_dir() and ((directory / 'report.md').is_file() or (directory / 'status.json').is_file()):
                entries.append(directory)
    entries.sort(key=lambda directory: directory.stat().st_mtime, reverse=True)
    if not entries:
        print('尚无任务。量化产物将保存在 runs/<任务>-quant/candidates/<方案>/quantized/。')
        return
    entries = entries[:40]
    for index, directory in enumerate(entries, 1):
        status = read_json(directory / 'status.json').get('status', '未知') if (directory / 'status.json').is_file() else '有报告'
        print(f'{index:2}  {directory.relative_to(root)}  [{status}]')
    answer = ask('输入编号查看报告和产物；回车返回')
    if not answer:
        return
    index = positive_integer(answer)
    if index > len(entries):
        raise ValueError('编号超出范围')
    directory = entries[index - 1]
    print('\n目录：' + str(directory))
    for name in ('report.md', 'report.html', 'search-details.md', 'search-details.html', 'selected_model.json', 'deployment/start.sh'):
        if (directory / name).exists():
            print(name + '：' + str(directory / name))
    if (directory / 'report.md').is_file():
        print('\n' + (directory / 'report.md').read_text())
    for receipt_path in sorted((directory / 'candidates').glob('*/model_manifest.json')):
        receipt = read_json(receipt_path)
        artifact = Path(receipt['artifact'])
        print(f'\n{receipt_path.parent.name} 实际权重：{artifact}' + ('' if artifact.is_dir() else '（当前路径不可访问）'))
        print('  加载视图：' + receipt['load_view'] + '（可能使用软链接，不要只复制此目录）')
    if (directory / 'request.json').is_file():
        request = read_json(directory / 'request.json')
        print('量化任务：' + str(request.get('quantization_run')) + '；加速任务：' + str(request.get('inference_run')))


def resume_arguments(root, run_id):
    directory = root / 'runs/workflows' / identifier(run_id)
    request = read_json(directory / 'request.json')
    invocation = request.get('invocation')
    if not invocation:
        raise ValueError('此任务没有保存新版菜单参数，请使用当时的原命令加 --resume 继续')
    if invocation.get('run_id') != run_id:
        raise ValueError('任务目录与记录的任务名称不一致')
    arguments = ['--resume']
    scalar = ('model', 'mtp_model', 'dataset', 'recipes', 'gpus', 'quantization_config', 'inference_config',
              'reuse_quantization_run', 'from_workflow', 'candidate', 'profile', 'run_id', 'max_model_len', 'max_jobs',
              'mode', 'selection_objective', 'calibration_count', 'calibration_tokens',
              'performance_repeats', 'screening_samples', 'search_trials', 'final_repeats')
    for key in scalar:
        if invocation.get(key) is not None:
            arguments += ['--' + key.replace('_', '-'), str(invocation[key])]
    for key in ('skip_quantization', 'auto_select'):
        if invocation.get(key):
            arguments.append('--' + key.replace('_', '-'))
    return arguments


def deploy(root):
    bundle = path_input(ask('部署包目录（包含deployment.json）'))
    receipt = read_json(bundle / 'deployment.json')
    if receipt.get('stage') != 'portable_deployment':
        raise ValueError('不是本工具生成的部署包')
    model = model_input(ask('模型目录', receipt.get('default_model')), True)
    if not model.is_dir():
        raise ValueError('启动服务需要真实模型目录，不能传入选择清单')
    arguments = ['--bundle', str(bundle), '--model', str(model)]
    if receipt['trial']['mtp']:
        draft = path_input(ask('MTP目录', receipt.get('default_mtp')))
        if not (draft / 'config.json').is_file():
            raise ValueError('MTP目录不可访问')
        arguments += ['--mtp-model', str(draft)]
    port = positive_integer(ask('服务端口', '9001'))
    if port > 65535:
        raise ValueError('端口应在1–65535之间')
    arguments += ['--port', str(port), '--host', ask('监听地址', '127.0.0.1')]
    gpu = normalize_gpus(ask('服务使用的单张GPU编号', '0'))
    if ',' in gpu:
        raise ValueError('本部署适配器使用单张GPU')
    arguments += ['--gpu', gpu]
    mode = ask('模式：1=请求验收后退出，2=持续服务', '1')
    if mode not in ('1', '2'):
        raise ValueError('模式请输入1或2')
    if mode == '1':
        arguments.append('--smoke-test')
    print('等价命令：' + shlex.join([str(root / 'quickstart'), 'deploy', *arguments]))
    if ask('开始？y/n', 'y').lower() == 'y':
        from .portable import main as portable_main
        portable_main(['--root', str(root), *arguments], install_signals=False)


def doctor(root):
    print('\n' + defaults_text())
    usage = shutil.disk_usage(root)
    print(f'工作目录：{root}\n可用磁盘：{usage.free / 1024**3:.1f} GiB')
    from .models import memory_limit_gib
    memory = memory_limit_gib()
    print('检测到的内存上限GiB：' + (f'{memory:.1f}' if memory is not None else '未检测到'))
    if Path('/dev/shm').exists():
        print(f'/dev/shm 可用：{shutil.disk_usage("/dev/shm").free / 1024**3:.1f} GiB')
    executable = shutil.which('nvidia-smi')
    if executable:
        result = subprocess.run([executable, '--query-gpu=index,name,memory.total,driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=20)
        print('GPU：\n' + (result.stdout.strip() if result.returncode == 0 else result.stderr.strip()))
    else:
        print('GPU：未发现nvidia-smi；可以浏览菜单，实际量化/推理需要NVIDIA GPU。')
    for stage in ('quantization', 'inference'):
        config = read_json(resolve_config(root, None, stage))
        python = config['backends']['llmcompressor']['python'] if stage == 'quantization' else config['runtime']['python']
        print(stage + ' Python：' + python)
        if not Path(python).is_file():
            print('  环境不存在；源码使用者需要先配置运行环境。')
            continue
        code = ('import json;from importlib.metadata import version;'
                'print(json.dumps({name:version(name) for name in ' + repr(
                    ['torch', 'llmcompressor', 'compressed-tensors'] if stage == 'quantization'
                    else ['torch', 'vllm', 'flashinfer-python']) + '}))')
        result = subprocess.run([python, '-c', code], capture_output=True, text=True, timeout=30)
        print('  ' + (result.stdout.strip() if result.returncode == 0 else '依赖检查失败：' + result.stderr.strip()[-1000:]))
    print('这里只检查环境和资源，不代表GPU推理已验收；完整流程会执行真实加载和请求。')
    print('持久保存：将 PIPELINE_WORKSPACE 指向挂载盘中的新目录；重建实例后重新挂载同一目录。')


def main(argv=None):
    parser = argparse.ArgumentParser(description=NAME + ' · ' + CHINESE_NAME + ' 交互入口')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--doctor', action='store_true')
    parser.add_argument('--workspace', action='store_true', help='更换镜像工作目录，保留原任务')
    parser.add_argument('--plain', action='store_true', help='使用兼容数字菜单')
    parser.add_argument('--version', action='version', version=NAME + ' ' + VERSION)
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    root = args.root.resolve()
    if args.workspace:
        from .storage import change_workspace
        try:
            return change_workspace(root, plain=args.plain)
        except (ValueError, OSError) as error:
            print(str(error))
            return 1
    if not args.plain and not args.doctor and sys.stdin.isatty() and os.environ.get('TERM') != 'dumb':
        try:
            import rich
            import prompt_toolkit
        except ImportError:
            print('增强交互依赖未安装，使用数字菜单；依赖见 requirements/terminal.txt。')
        else:
            from .guided import menu as guided_menu
            return guided_menu(root)
    banner(root)
    if args.doctor:
        doctor(root)
        return
    if not sys.stdin.isatty():
        print('菜单需要交互终端。Docker请使用 -it；自动任务可用 quickstart workflow --model ...。')
        return
    while True:
        menu()
        try:
            choice = ask('选择', '0')
            if choice == '0':
                print('已退出 InferBoost；已有产物保留。')
                return
            if choice in ('1', '2'):
                wizard = WorkflowWizard(root, choice == '2')
                while run_workflow(root, wizard.collect(), allow_back=True) == 'back':
                    wizard.index = len(wizard.questions) - 1
            elif choice == '3':
                history(root)
            elif choice == '4':
                run_workflow(root, resume_arguments(root, ask('完整流程任务名称')))
            elif choice == '5':
                deploy(root)
            elif choice == '6':
                doctor(root)
            elif choice == '7':
                from .storage import change_workspace
                change_workspace(root, plain=True)
            else:
                print('请输入菜单中的编号。')
        except WizardCancelled:
            terminal.message('已返回主菜单；未创建任务。', 'muted')
        except EOFError:
            print('\n输入已关闭，任务记录保留。')
            return
        except KeyboardInterrupt:
            print('\n操作已中断，原始日志和产物保留。')
        except (ValueError, OSError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
            terminal.message('未完成：' + str(error), 'error')


if __name__ == '__main__':
    main()
