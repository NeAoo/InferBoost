"""Run bounded search phases through the official vLLM sweep."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import time

from pipeline.common import digest, identifier, read_json, write_json
from pipeline.quality import load_dataset
from pipeline.resources import gpu_lease
from pipeline.terminal import Progress, section
from . import assets, planning
from .config import ADAPTER_VERSION, environment


@contextmanager
def gpu_lock(root, wait_seconds=0, run=None, gpu=0):
    with gpu_lease(root, gpu, wait_seconds):
        yield


def snapshot(root, destination):
    files = list((root / 'pipeline').rglob('*.py'))
    files.append(root / 'scripts/prepare_qwen_assets.py')
    hashes = {}
    for path in files:
        if '__pycache__' in path.parts:
            continue
        relative = path.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = digest(target)
    return hashes


def phase_results(run, phase=None):
    directory = run / 'official' / phase if phase else run / 'official'
    results = []
    if directory.exists():
        for path in directory.rglob('run=*.json'):
            if not re.fullmatch(r'run=\d+\.json', path.name):
                continue
            data = read_json(path)
            data['_file'] = str(path)
            results.append(data)
    return results


def complete(result, samples):
    return result.get('stage2_status') == 'complete' and result.get('samples') == samples and all(
        values['requests'] == samples and values['successes'] == samples
        and values['errors'] == 0 and values['truncated'] == 0
        for values in result.get('cache', {}).values()) and set(result.get('cache', {})) == {'cold', 'warm'}


def screen_ranking(run, config):
    candidates = read_json(run / 'candidates.json')
    measured = {}
    cache = config['search']['objective_cache']
    for result in phase_results(run):
        phase = Path(result['_file']).relative_to(run / 'official').parts[0]
        spec = read_json(run / 'phases' / (phase + '.json'))
        if spec['kind'] == 'screen' and complete(result, spec['samples']):
            measured[result['profile_id']] = result['cache'][cache]
    baseline_id = read_json(run / 'plan.json')['baseline']['id']
    reference = measured.get(baseline_id, {}).get('quality_score')
    if reference is not None:
        measured = {key: value for key, value in measured.items()
                    if value['quality_score'] is not None and value['quality_score'] + 1e-12 >= reference}
    return [candidates[key] for key in sorted(measured, key=lambda key: (measured[key]['e2e_s'], key))]


def publish_phase(run, config, model, name):
    from . import search_report
    from .text_report import stage_markdown
    from pipeline.quantization_labels import describe_quantization
    candidates = read_json(run / 'candidates.json')
    evidence = search_report.collect(run, config, candidates, read_json(run / 'plan.json'))
    recipe = config.get('handoff', {}).get('quantization', {}).get('recipe')
    label = describe_quantization(recipe, precision=model['precision'])['quantization_label']
    search_report.render(run, evidence, label)
    step = next((step for step in evidence['decision_steps'] if step['phase'] == name), None)
    print('\n' + (stage_markdown(step) if step else '阶段完成：' + name + '；详细结果见筛选报告。'), flush=True)
    print('阶段报告：' + str(run / 'search-details.md') + '\nHTML：' + str(run / 'search-details.html'), flush=True)


PHASE_LABELS = {'attention': '注意力后端', 'kernels': '量化矩阵内核', 'graphs': '编译与 CUDA Graph',
                'budgets': '调度预算', 'mtp': 'MTP', 'final': '完整复测',
                'full-screen': '全部组合', 'baseline-fallback': '基础组合补测'}


def wait_with_progress(process, run, name, interval=2):
    """Display actual request-pair counts; loading/compile time stays indeterminate."""
    phase_label = PHASE_LABELS.get(name, name)
    display = Progress('加速 · ' + phase_label)
    scope, code = None, None
    try:
        display.update(detail='准备服务；日志 phases/' + name + '.log', force=True)
        while True:
            try:
                code = process.wait(timeout=interval)
                return code
            except subprocess.TimeoutExpired:
                try:
                    path = run / 'progress.json'
                    progress = read_json(path) if path.is_file() else {}
                    active_path = run / 'active.json'
                    active = read_json(active_path) if active_path.is_file() else {}
                except (OSError, ValueError):
                    continue
                valid = (progress.get('phase') == name and (not active or (
                         active.get('phase') == name and (not progress.get('profile_id')
                         or progress.get('profile_id') == active.get('profile_id')))))
                if not valid:
                    if scope is not None:
                        display.close('进入下一个配置')
                        display = Progress('加速 · ' + phase_label)
                        scope = None
                    display.update(detail='加载/编译/预热；详见阶段日志')
                    continue
                key = (progress.get('profile_id', progress.get('profile')), progress.get('round'), progress.get('stage', 'performance'))
                if scope != key:
                    display.close('进入下一项')
                    stage = {'performance': '性能请求', 'quality': '质量复核', 'warmup': '预热'}.get(key[-1], '准备')
                    display = Progress(phase_label + ' · ' + stage, progress.get('total'), unit='组cold+warm')
                    print('配置：' + progress['profile'] + ' / ' + progress['round'], flush=True)
                    scope = key
                display.update(progress.get('completed', 0))
    finally:
        display.close('阶段进程结束' if code == 0 else '已中断' if code is None else '失败')


def run_phase(root, run, config, model, name, trials, kind, samples, repeats):
    candidates_file = run / 'candidates.json'
    all_candidates = read_json(candidates_file) if candidates_file.exists() else {}
    phase_path = run / 'phases' / (name + '.json')
    if phase_path.exists():
        phase = read_json(phase_path)
        trials = phase['candidates']
    else:
        if kind == 'screen':
            trials = [trial for trial in trials if trial['id'] not in all_candidates]
            available = config['search']['max_trials'] - len(all_candidates)
            trials = trials[:max(0, available)]
        if not trials:
            print(f'[{name}] 没有新增可执行配置；复用已测结果或查看兼容性跳过说明。', flush=True)
            return
        for trial in trials:
            trial['command'] = planning.server_command(config, model, trial)
            all_candidates[trial['id']] = trial
        phase = {'name': name, 'kind': kind, 'samples': samples, 'repeats': repeats, 'candidates': trials}
        write_json(candidates_file, all_candidates)
        write_json(phase_path, phase)
    output = run / 'official' / name
    records = [{'_benchmark_name': trial['id'], 'profile_id': trial['id']} for trial in trials]
    parameters = run / 'phases' / (name + '.serve-params.json')
    write_json(parameters, records)
    python = config['runtime']['python']
    serve = [python, '-m', 'pipeline.inference.server', '--run-dir', str(run), '--phase', name,
             '--host', '127.0.0.1', '--port', str(config['server']['port'])]
    bench = [python, '-m', 'pipeline.inference.worker', '--run-dir', str(run)]
    reset = [python, '-m', 'pipeline.inference.worker', 'reset', '--run-dir', str(run)]
    command = [python, '-m', 'pipeline.inference.native', '--serve-cmd', shlex.join(serve),
               '--bench-cmd', shlex.join(bench), '--after-bench-cmd', shlex.join(reset),
               '--serve-params', str(parameters), '--output-dir', str(run / 'official'),
               '--experiment-name', name, '--num-runs', str(repeats), '--continue-on-error',
               '--show-stdout', '--server-ready-timeout', str(config['server']['startup_timeout_seconds'])]
    if output.exists():
        # Native sweep writes a summary even when only some repetitions succeeded.
        # Preserve that file, but let native resume retry missing repetitions.
        for summary in output.rglob('summary.json'):
            recorded = read_json(summary)
            if isinstance(recorded, list) and len(recorded) < repeats:
                summary.rename(summary.with_name('summary.partial-' + str(time.time_ns()) + '.json'))
        command.append('--resume')
    write_json(run / 'status.json', {'status': 'searching', 'phase': name, 'candidates': len(trials),
                                     'samples': samples, 'repeats': repeats, 'pid': os.getpid()})
    write_json(run / 'phases' / (name + '.command.json'), {'command': command, 'engine': 'vllm.benchmarks.sweep.serve'})
    env = environment(root, config, run / 'source')
    section('加速 · ' + PHASE_LABELS.get(name, name),
            f'{len(trials)} 个配置；每种缓存 {samples} 条 × {repeats} 轮；日志 phases/{name}.log')
    with (run / 'phases' / (name + '.log')).open('a') as log:
        process = subprocess.Popen(command, cwd=run / 'source', env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = wait_with_progress(process, run, name)
        except BaseException:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        raise RuntimeError(f'官方sweep执行失败({code})，查看 phases/{name}.log')
    if kind == 'screen':
        publish_phase(run, config, model, name)


def run(root, config, model_path, mtp_path, run_id, resume=False):
    from .defaults import defaults_text
    print('\n' + defaults_text(), flush=True)
    run = root / 'runs/inference' / identifier(run_id)
    fingerprint = hashlib.sha256(json.dumps({'config': config, 'model': str(Path(model_path).resolve()),
                                            'mtp': str(mtp_path) if mtp_path else None,
                                            'dataset': digest(config['benchmark']['dataset']),
                                            'quality_dataset': digest(config['quality']['dataset']) if config.get('quality', {}).get('dataset') else None,
                                            'adapter': ADAPTER_VERSION,
                                            'code': {path.name: digest(path) for path in sorted((root / 'pipeline/inference').glob('*.py'))}},
                                           sort_keys=True).encode()).hexdigest()
    if resume:
        if read_json(run / 'manifest.json')['signature'] != fingerprint:
            raise ValueError('配置或输入已变化，不能混入旧结果；请新建run-id')
    else:
        run.mkdir(parents=True, exist_ok=False)
        config = dict(config, root=str(root))
        write_json(run / 'config.json', config)
        write_json(run / 'manifest.json', {'signature': fingerprint, 'adapter': ADAPTER_VERSION,
                                          'started_at': time.time(), 'stage': 2, 'requantization': False})
        from pipeline.modes import sample_rows
        rows = sample_rows(load_dataset(config['benchmark']['dataset']), config['benchmark'].get('sample_limit'))
        write_json(run / 'inputs.json', rows)
        if config.get('quality', {}).get('dataset'):
            write_json(run / 'quality_inputs.json', sample_rows(load_dataset(config['quality']['dataset']),
                                                               config['quality'].get('sample_limit')))
        write_json(run / 'source_manifest.json', snapshot(root, run / 'source'))
    config = read_json(run / 'config.json')
    try:
        with gpu_lock(root, config['search'].get('gpu_wait_seconds', 0), run, config['runtime'].get('gpu', 0)):
            from .server import assert_idle
            assert_idle(config, wait_seconds=1)
            hardware = assets.probe(root, config)
            if resume and read_json(run / 'hardware.json')['uuid'] != hardware['uuid']:
                raise ValueError('GPU已更换，不能续跑同一比较批次')
            write_json(run / 'hardware.json', hardware)
            if (run / 'model.json').exists():
                model = read_json(run / 'model.json')
                assets.verify_model(model)
            else:
                model = assets.prepare(root, run, config, model_path, mtp_path)
            if config.get('handoff'):
                expected = config['handoff']['identity']['shards']
                actual = model['identity']['shards']
                if set(expected) != set(actual) or any(expected[name]['sha256'] != actual[name]['sha256'] for name in expected):
                    raise ValueError('模型分片与第一阶段选定清单不一致')
                write_json(run / 'handoff.json', config['handoff'])
            capability = planning.capabilities(model, hardware, config)
            base = planning.baseline(capability, config)
            write_json(run / 'plan.json', {'capabilities': capability, 'baseline': base,
                                           'search': config['search'], 'precision': model['precision'],
                                           'native_sweep': True, 'compile_and_graph_independent': True})
            rows = read_json(run / 'inputs.json')
            count = min(config['search']['screening_samples'], len(rows))
            if config['search']['strategy'] == 'full':
                trials = planning.full_grid(capability, config)
                run_phase(root, run, config, model, 'full-screen', trials, 'screen', count, 1)
            else:
                parents = [base]
                for name in ['attention', 'kernels', 'graphs', 'budgets', 'mtp']:
                    trials = planning.screen_phase(name, parents, capability, config)
                    run_phase(root, run, config, model, name, trials, 'screen', count, 1)
                    ranked = screen_ranking(run, config)
                    if not ranked and name == 'attention':
                        fallback = [planning.replace(base, attention=a, kernel=k)
                                    for a in capability['allowed']['attention'] for k in capability['allowed']['kernels']]
                        run_phase(root, run, config, model, 'baseline-fallback', fallback, 'screen', count, 1)
                        ranked = screen_ranking(run, config)
                    if not ranked:
                        raise RuntimeError('尚无通过完整请求校验的基础配置，详见阶段日志')
                    parents = ranked[:config['search']['finalists']]
            ranked = screen_ranking(run, config)
            if not ranked:
                raise RuntimeError('没有完整的筛选结果')
            all_candidates = read_json(run / 'candidates.json')
            baseline_id = base['id'] if base['id'] in {trial['id'] for trial in ranked} else ranked[0]['id']
            finalists = planning.unique([all_candidates[baseline_id], *ranked[:config['search']['finalists']]])
            write_json(run / 'finalists.json', {'baseline_id': baseline_id, 'profiles': [trial['id'] for trial in finalists]})
            run_phase(root, run, config, model, 'final', finalists, 'final', len(rows), config['search']['repeats'])
            from .report import generate
            report = generate(run)
            from .text_report import final_markdown
            from pipeline.text_report import report_paths
            print(final_markdown(report, read_json(run / 'candidates.json')), flush=True)
            report_paths(run, search=True)
            state = 'completed' if report['recommendation'] else 'completed_without_recommendation'
            write_json(run / 'status.json', {'status': state, 'finished_at': time.time(),
                                             'report': str(run / 'report.html')})
            return run
    except BaseException as error:
        write_json(run / 'status.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                         'error': str(error), 'time': time.time()})
        raise
