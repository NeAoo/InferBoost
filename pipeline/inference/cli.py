"""Independent stage-two entrypoint; accepts a model or stage-one selected_model.json."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys

from pipeline.common import digest, identifier, read_json
from . import assets, config as settings, planning


def resolve_model(path, mtp=None):
    path = Path(path).expanduser().resolve()
    handoff = None
    if path.is_file():
        handoff = read_json(path)
        if handoff.get('schema_version') != 1 or handoff.get('stage') != 'quantization_selection':
            raise ValueError('只接受模型目录或第一阶段 selected_model.json')
        selection = handoff.get('selection', {})
        user_choice = selection.get('mode') == 'user' and selection.get('quality_criteria_are_advisory') is True
        if not handoff.get('quality_gate', {}).get('passed') and not user_choice:
            raise ValueError('第一阶段候选未达到推荐条件，且没有用户选择记录')
        candidate = identifier(handoff['candidate'])
        receipt_path = path.parent / 'candidates' / candidate / 'model_manifest.json'
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            if digest(Path(handoff['model']) / 'config.json') != receipt['load_config_sha256']:
                raise ValueError('第一阶段选定加载配置已变化')
        mtp = mtp or handoff.get('mtp_source')
        path = Path(handoff['model']).resolve()
    if not (path / 'config.json').is_file():
        raise ValueError('找不到可加载模型目录: ' + str(path))
    return path, Path(mtp).resolve() if mtp else None, handoff


def default_config(root):
    for name in ['configs/inference.local.json', 'configs/inference.example.json']:
        if (root / name).is_file():
            return root / name
    raise ValueError('请先配置 configs/inference.example.json')


def stop(signum, frame):
    raise KeyboardInterrupt('signal ' + str(signum))


def main():
    argv = sys.argv[1:]
    if len(argv) > 2 and argv[0] == '--root' and argv[2].startswith('--') and argv[2] != '--help':
        argv.insert(2, 'run')
    parser = argparse.ArgumentParser(description='第二阶段：已有模型的推理加速搜索与部署')
    parser.add_argument('--root', type=Path, required=True)
    commands = parser.add_subparsers(dest='command')
    for name in ['inspect', 'plan', 'run']:
        command = commands.add_parser(name)
        command.add_argument('--model', type=Path, required=True, help='模型目录或第一阶段 selected_model.json')
        command.add_argument('--mtp-model', type=Path)
        command.add_argument('--config', type=Path)
        command.add_argument('--run-id', default='inference')
        command.add_argument('--strategy', choices=['staged', 'full'])
        command.add_argument('--sm', type=int, help='仅用于离线plan，例如80/90')
        if name == 'run':
            command.add_argument('--resume', action='store_true')
            command.add_argument('--wait-for-gpu', type=int, default=0, help='等待其他阶段释放GPU的最大秒数')
    for name in ['report', 'status', 'select', 'serve']:
        command = commands.add_parser(name)
        command.add_argument('--run-id', required=True)
        if name in ('select', 'serve'):
            command.add_argument('--profile', default='recommended' if name == 'select' else None)
        if name == 'select':
            command.add_argument('--cache', choices=['cold', 'warm'])
        if name == 'serve':
            command.add_argument('--host', default='127.0.0.1')
            command.add_argument('--port', type=int)
            command.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    signal.signal(signal.SIGTERM, stop)
    if args.command is None:
        model = input('量化模型目录，或第一阶段 selected_model.json：').strip()
        config = input('推理环境配置 [' + str(default_config(root)) + ']：').strip()
        run_id = input('本次运行ID [inference]：').strip() or 'inference'
        sys.argv = [sys.argv[0], '--root', str(root), 'run', '--model', model, '--run-id', run_id]
        if config:
            sys.argv += ['--config', config]
        return main()
    if args.command in ('inspect', 'plan', 'run'):
        model_path, mtp, handoff = resolve_model(args.model, args.mtp_model)
        if args.command == 'inspect':
            result = assets.inspect(model_path)
            result.pop('tensor_headers')
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        config = settings.load(root, args.config or default_config(root))
        if handoff:
            config['handoff'] = handoff
        if args.strategy:
            config['search']['strategy'] = args.strategy
        if args.command == 'plan':
            model = assets.inspect(model_path)
            model['mtp_supported'] = bool(mtp) or model['bf16_mtp_present']
            hardware = assets.probe(root, config) if args.sm is None else {
                'capability': [args.sm // 10, args.sm % 10], 'humming': True, 'flashinfer': True,
                'scope': 'user_declared_offline_plan_not_runtime_verified'}
            capability = planning.capabilities(model, hardware, config)
            result = {'capabilities': capability, 'baseline': planning.baseline(capability, config),
                      'precision': model['precision'], 'search': config['search'], 'offline': args.sm is not None}
            if config['search']['strategy'] == 'full':
                result['trials'] = planning.full_grid(capability, config)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.sm is not None:
            raise ValueError('--sm仅用于离线计划，实际执行必须检测真实GPU')
        if args.wait_for_gpu < 0:
            raise ValueError('GPU等待秒数不能为负数')
        config['search']['gpu_wait_seconds'] = args.wait_for_gpu
        from .execution import run
        result = run(root, config, model_path, mtp, args.run_id, args.resume)
        return 0
    run = root / 'runs/inference' / identifier(args.run_id)
    if args.command == 'report':
        from .report import generate
        report = generate(run)
        from .text_report import final_markdown
        from pipeline.text_report import report_paths
        print(final_markdown(report, read_json(run / 'candidates.json')), flush=True)
        report_paths(run, search=True)
    elif args.command == 'status':
        print(json.dumps({'status': read_json(run / 'status.json'),
                          'progress': read_json(run / 'progress.json') if (run / 'progress.json').exists() else None},
                         ensure_ascii=False, indent=2))
    elif args.command == 'select':
        from .deploy import select
        print('部署方案：' + str(select(run, args.profile, args.cache)))
    elif args.command == 'serve':
        from .deploy import serve
        serve(root, run, args.profile, args.host, args.port, args.smoke_test)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('推理加速任务已中断，原始记录保留。', file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.SubprocessError) as error:
        print('INFERENCE_ERROR: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
