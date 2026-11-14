"""One entry point joining measured quantization, tuning and portable deployment."""
import argparse
import copy
import json
from pathlib import Path
import signal
import sys
import time

from .common import digest, identifier, read_json, source_digest, write_json
from .calibration import official_defaults
from .terminal import section, message
from .quantization_catalog import FRESH_RECIPES, SCHEMES


RECIPE_NAMES = {name.removeprefix('compressor_'): name for name in FRESH_RECIPES}


def resolve_config(root, explicit, stage):
    if explicit:
        return Path(explicit).expanduser().resolve()
    names = (['quantization-llmcompressor.local.json', 'quantization.example.json']
             if stage == 'quantization' else ['inference.local.json', 'inference.example.json'])
    for name in names:
        path = root / 'configs' / name
        if path.is_file():
            return path
    raise ValueError('缺少运行环境配置: ' + stage)


def choose_recipes(value):
    names = list(RECIPE_NAMES) if value == 'all' else list(dict.fromkeys(value.split(',')))
    if not names:
        raise ValueError('至少选择一种量化配方')
    result = []
    for name in names:
        name = name.strip()
        recipe = RECIPE_NAMES.get(name, name)
        if recipe not in FRESH_RECIPES:
            raise ValueError('未知量化配方: ' + name)
        result.append({'id': recipe.removeprefix('compressor_'), 'recipe': recipe, 'mode': 'fresh'})
    return result


def reuse_exports(root, quantization, run_id):
    previous = root / 'runs' / identifier(run_id)
    previous_config = read_json(previous / 'config.json')
    if Path(previous_config['source']).resolve() != Path(quantization['source']).resolve():
        raise ValueError('复用批次的原模型目录不同；不能混用来源')
    calibration = Path(read_json(previous / 'calibration_manifest.json')['path'])
    quantization['calibration'] = {'mode': 'frozen', 'path': str(calibration)}
    old_candidates = {item['id']: item for item in previous_config['candidates']}
    for item in quantization['candidates']:
        if item['recipe'] == 'original':
            continue
        old = old_candidates.get(item['id'])
        receipt_path = previous / 'candidates' / item['id'] / 'model_manifest.json'
        if not old or old['recipe'] != item['recipe'] or not receipt_path.is_file():
            continue
        receipt = read_json(receipt_path)
        artifact = Path(receipt['artifact'])
        if not (artifact / 'quantization_ready.json').is_file() or receipt.get('mtp_unchanged') is not True:
            raise ValueError('候选没有完整的导出及MTP检查记录: ' + item['id'])
        # prepare_view rechecks source identity, shard payloads, precision and MTP
        # before any reused export can be measured in the new batch.
        item.update(mode='reuse', model=str(artifact), reused_from_run=run_id)


def build_configs(root, args):
    if getattr(args, 'from_workflow', None):
        from .retest import configs
        return configs(args)
    for name in ['max_jobs', 'max_model_len']:
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(name + ' 必须为正整数')
    quantization = copy.deepcopy(read_json(resolve_config(root, args.quantization_config, 'quantization')))
    inference = copy.deepcopy(read_json(resolve_config(root, args.inference_config, 'inference')))
    quantization['source'] = str(args.model.expanduser().resolve())
    quantization['mtp_source'] = str(args.mtp_model.expanduser().resolve()) if args.mtp_model else None
    from .modes import MODES
    recipes = args.recipes or ','.join(MODES[getattr(args, 'mode', 'custom')]['recipes'])
    quantization['candidates'] = [{'id': 'original', 'recipe': 'original', 'mode': 'reuse'}, *choose_recipes(recipes)]
    offline_source = quantization.get('calibration', {}).get('raw_data')
    quantization['calibration'] = official_defaults()
    if offline_source:
        quantization['calibration']['raw_data'] = str((root / offline_source).resolve())
    quantization['datasets'] = [{'id': 'default_quality100', 'path': str(root / 'data/quality100.json')}]
    inference['benchmark']['dataset'] = str(root / 'data/public100.json')
    inference['quality'] = {'dataset': str(root / 'data/quality100.json')}
    if args.dataset:
        from .quality import load_dataset
        custom = str(args.dataset.expanduser().resolve())
        load_dataset(custom)
        quantization['datasets'].append({'id': 'custom', 'path': custom})
        # Keep built-in and custom stage-one scores separate. Stage two tunes
        # the user's declared workload and checks its labelled rows if supplied.
        inference['benchmark']['dataset'] = custom
        inference['quality']['dataset'] = custom
    devices = [int(value) for value in args.gpus.split(',')]
    if not devices or len(devices) != len(set(devices)) or min(devices) < 0:
        raise ValueError('--gpus 需要不重复的GPU编号，例如 0,1')
    quantization['runtime'].update(gpu=devices[0], quantization_gpus=devices)
    inference['runtime']['gpu'] = devices[0]
    if args.max_model_len:
        for settings in (quantization, inference):
            settings['server']['max_model_len'] = args.max_model_len
    if args.max_jobs:
        for settings in (quantization, inference):
            settings['runtime']['max_jobs'] = args.max_jobs
    if not args.mtp_model:
        inference['search']['mtp_depths'] = [0]
    from .modes import apply_mode
    apply_mode(quantization, inference, args)
    if getattr(args, 'reuse_quantization_run', None):
        if args.skip_quantization:
            raise ValueError('--reuse-quantization-run 与 --skip-quantization 不能同时使用')
        reuse_exports(root, quantization, args.reuse_quantization_run)
    return quantization, inference


def resume_contract(root, args, quantization, inference):
    data_paths = {item['path'] for item in quantization['datasets']}
    data_paths.update([inference['benchmark']['dataset'], inference['quality']['dataset']])
    calibration = quantization['calibration']
    if args.skip_quantization:
        data_paths = {inference['benchmark']['dataset'], inference['quality']['dataset']}
    elif calibration.get('mode') == 'frozen':
        directory = root / calibration['path']
        data_paths.update(str(directory / name) for name in ['manifest.json', 'cases.jsonl', 'tokens.jsonl'])
    elif calibration.get('raw_data'):
        data_paths.add(calibration['raw_data'])
    return {
        'model': str(args.model.resolve()),
        'mtp': str(args.mtp_model.resolve()) if args.mtp_model else None,
        'skip_quantization': args.skip_quantization,
        'quantization': quantization, 'inference': inference,
        'data_sha256': {path: digest(path) for path in sorted(data_paths)},
        'source_sha256': source_digest(root),
    }


def verify_resume(root, directory, args):
    request = read_json(directory / 'request.json')
    quantization, inference = build_configs(root, args)
    expected = request.get('resume_contract')
    actual = resume_contract(root, args, quantization, inference)
    if expected != actual:
        raise ValueError('续跑的模型/MTP、配置、数据或代码已变化；请保持原参数或使用新的run-id')
    if (read_json(directory / 'quantization.json') != expected['quantization']
            or read_json(directory / 'inference.json') != expected['inference']):
        raise ValueError('冻结的流程配置已修改，不能继续使用旧结果')


def run(root, args):
    from . import quantization
    from .inference import config as inference_settings
    from .inference.cli import resolve_model
    from .inference.execution import run as accelerate
    from .inference.deploy import select as select_profile
    from .portable import export_bundle, serve_bundle

    run_id = identifier(args.run_id)
    directory = root / 'runs/workflows' / run_id
    quant_id = run_id + '-quant'
    inference_id = run_id + '-accel'
    identifier(quant_id)
    identifier(inference_id)
    if not args.resume:
        if (root / 'runs/inference' / inference_id).exists():
            raise ValueError('加速任务名称已存在，请更换run-id；不能复用另一模型的结果')
        directory.mkdir(parents=True, exist_ok=False)
        quant_config, inference_config = build_configs(root, args)
        write_json(directory / 'quantization.json', quant_config)
        write_json(directory / 'inference.json', inference_config)
        write_json(directory / 'request.json', {
            'model': str(args.model.resolve()), 'skip_quantization': args.skip_quantization,
            'mtp': str(args.mtp_model.resolve()) if args.mtp_model else None,
            'auto_select_authorized': args.auto_select, 'started_at': time.time(),
            'invocation': {name: str(value.expanduser().resolve()) if isinstance(value, Path) else value
                           for name, value in vars(args).items() if name != 'root'},
            'quantization_run': quant_id, 'inference_run': inference_id,
            'retest_source': inference_config.get('retest_source'),
            'resume_contract': resume_contract(root, args, quant_config, inference_config),
        })
    else:
        verify_resume(root, directory, args)
    try:
        selected = args.model
        if getattr(args, 'from_workflow', None):
            from .retest import selection
            message('正在核验所选权重、加载配置与MTP记录；大模型校验需要读取权重文件。')
            selected = directory / 'selected_model.json'
            write_json(selected, selection(args))
            message('复用已完成的量化模型；新建加速对比，保留原任务与部署包。', 'success')
        if not args.skip_quantization:
            section('流程 1/3 · 量化与统一评测', '原模型参照 → 兼容配方 → 质量与速度 → 选择模型')
            write_json(directory / 'status.json', {'status': 'quantization', 'run': quant_id})
            quant_dir = root / 'runs' / quant_id
            status = read_json(quant_dir / 'status.json') if (quant_dir / 'status.json').is_file() else {}
            if status.get('status') not in ('completed', 'completed_with_failures'):
                quantization.run(root, directory / 'quantization.json', quant_id,
                                 resume=args.resume and quant_dir.exists())
            from .quantization_report import generate
            report = generate(quant_dir)
            from .text_report import quantization_summary, report_paths
            if status.get('status') in ('completed', 'completed_with_failures'):
                print(quantization_summary(report), flush=True)
            report_paths(quant_dir)
            choice = args.candidate
            if not choice and args.auto_select:
                choice = 'recommended'
            if not choice and sys.stdin.isatty():
                choice = input('选择模型ID，recommended=推荐，回车只保存报告：').strip()
            if not choice:
                write_json(directory / 'status.json', {'status': 'awaiting_model_selection', 'report': str(quant_dir / 'report.html')})
                return directory
            handoff = quantization.select(root, quant_id, choice)
            if args.auto_select and not args.candidate:
                handoff['selection'].update(mode='authorized_automatic', policy='recommendation_with_quality_criteria')
                write_json(quant_dir / 'selected_model.json', handoff)
            selected = quant_dir / 'selected_model.json'
        model, mtp, handoff = resolve_model(selected, args.mtp_model)
        inference = inference_settings.load(root, directory / 'inference.json')
        if handoff:
            inference['handoff'] = handoff
        section('流程 1/2 · 推理加速' if args.skip_quantization else '流程 2/3 · 推理加速',
                '注意力 → 矩阵内核 → 编译/Graph → 调度预算 → MTP → 完整复测')
        write_json(directory / 'status.json', {'status': 'acceleration', 'run': inference_id})
        acceleration_dir = root / 'runs/inference' / inference_id
        status = read_json(acceleration_dir / 'status.json') if (acceleration_dir / 'status.json').is_file() else {}
        if status.get('status') not in ('completed', 'completed_without_recommendation'):
            accelerate(root, inference, model, mtp, inference_id, resume=args.resume and acceleration_dir.exists())
        from .text_report import report_paths
        report_paths(acceleration_dir, search=True)
        profile = args.profile or ('recommended' if args.auto_select else None)
        if not profile and sys.stdin.isatty():
            profile = input('选择加速方案ID，recommended=推荐，回车只保存报告：').strip()
        if not profile:
            write_json(directory / 'status.json', {'status': 'awaiting_profile_selection', 'report': str(acceleration_dir / 'report.html')})
            return directory
        select_profile(acceleration_dir, profile)
        section('流程 2/2 · 导出与部署验收' if args.skip_quantization else '流程 3/3 · 导出与部署验收')
        bundle = export_bundle(acceleration_dir, directory / 'deployment')
        write_json(directory / 'status.json', {'status': 'deployment_exported', 'deployment': str(bundle)})
        serve_bundle(root, bundle, directory / 'inference.json', model=model, mtp=mtp,
                     smoke=True, port=inference['server']['port'])
        write_json(directory / 'status.json', {'status': 'completed', 'deployment': str(bundle),
                   'deployment_smoke_passed': True, 'finished_at': time.time(),
                   'quantization_report': str(root / 'runs' / quant_id / 'report.html') if not args.skip_quantization else None,
                   'acceleration_report': str(acceleration_dir / 'report.html')})
        message('完整流程通过；部署入口：' + str(bundle / 'start.sh'), 'success', bold=True)
        return directory
    except BaseException as error:
        write_json(directory / 'status.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   'error': str(error), 'updated_at': time.time()})
        raise


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description='输入模型 → 量化对比 → 选择 → 加速搜索 → 一键部署')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--mtp-model', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--recipes', help='默认按mode；也可传all或逗号分隔配方')
    parser.add_argument('--gpus', default='0', help='量化设备列表；性能统一在第一张卡测量')
    parser.add_argument('--quantization-config', type=Path)
    parser.add_argument('--inference-config', type=Path)
    parser.add_argument('--skip-quantization', action='store_true')
    parser.add_argument('--reuse-quantization-run', help='新建批次，复用指定量化run的完整导出并重新评分')
    parser.add_argument('--from-workflow', type=Path,
                        help='返回已完成量化阶段重新选模型；传原runs/workflows/任务目录，配合--candidate和新--run-id')
    parser.add_argument('--auto-select', action='store_true', help='明确授权按报告建议自动选模型及加速方案')
    parser.add_argument('--candidate')
    parser.add_argument('--profile')
    parser.add_argument('--run-id', default='pipeline-' + time.strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-model-len', type=int)
    parser.add_argument('--max-jobs', type=int)
    parser.add_argument('--mode', choices=['quick', 'standard', 'custom'], default='custom')
    parser.add_argument('--selection-objective', choices=['quality', 'latency', 'size'], default='size')
    for option in ['calibration-count', 'calibration-tokens', 'performance-repeats',
                   'screening-samples', 'search-trials', 'final-repeats']:
        parser.add_argument('--' + option, type=int)
    return parser.parse_args(argv)


def main(argv=None, install_signals=True):
    args = parse_arguments(argv)
    if args.from_workflow:
        overrides = ['reuse_quantization_run', 'dataset', 'inference_config', 'quantization_config',
                     'recipes', 'max_jobs', 'max_model_len', 'calibration_count', 'calibration_tokens',
                     'performance_repeats', 'screening_samples', 'search_trials', 'final_repeats']
        if any(getattr(args, key) is not None for key in overrides) or args.mode != 'custom' or args.gpus != '0':
            raise ValueError('--from-workflow 继承原测试配置，不可同时覆盖配置、数据或搜索预算')
        from .retest import configure
        configure(args)
    if not args.model:
        if not sys.stdin.isatty():
            raise ValueError('非交互运行需要 --model；查看 --help')
        args.model = Path(input('模型目录（完整精度或已量化）：').strip()).expanduser()
        settings = read_json(args.model / 'config.json')
        args.skip_quantization = bool(settings.get('quantization_config'))
        answer = input('MTP草稿目录，没有则回车；与主模型同目录可输入 same：').strip()
        args.mtp_model = args.model if answer == 'same' else Path(answer).expanduser() if answer else None
        if not args.skip_quantization:
            print('量化后端：LLM Compressor；配方：' + ', '.join(f'{key}={SCHEMES[value]}' for key, value in RECIPE_NAMES.items()))
            args.recipes = input('量化配方，逗号分隔 [all]：').strip() or 'all'
        answer = input('自定义测试数据JSON/JSONL，没有则回车：').strip()
        args.dataset = Path(answer).expanduser() if answer else None
        args.gpus = input('量化GPU编号，例如0或0,1 [0]：').strip() or '0'
        args.auto_select = input('按推荐自动走完全流程？y/n [n]：').strip().lower() == 'y'
    if install_signals:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    return run(args.root.resolve(), args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print('WORKFLOW_ERROR: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
