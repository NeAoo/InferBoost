"""Relocatable deployment recipes with measured parameters and content verification."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .benchmark import endpoint, request_one
from .common import digest, read_json, write_json
from .models import inspect_model
from .resources import gpu_lease


def portable_identity(identity):
    return {**{key: value for key, value in identity.items() if key != 'shards'},
            'shards': {name: {key: value for key, value in shard.items() if key != 'resolved_path'}
                       for name, shard in identity['shards'].items()}}


def export_bundle(run, destination):
    from .inference.report import generate
    run, destination = Path(run), Path(destination)
    report = generate(run)
    trial = read_json(run / 'selected.json')['profile']
    canonical = read_json(run / 'candidates.json').get(trial['id'])
    if canonical != trial:
        raise ValueError('选定配置与实测候选不一致，不能导出修改后的参数')
    if (trial['id'] not in {row['profile'] for row in report['rows']}
            or not report['quality_gates'].get(trial['id']) or not report['quality_comparison_ready']):
        raise ValueError('只能导出完整复测且通过质量复核的方案')
    destination.mkdir(parents=True, exist_ok=True)
    model = read_json(run / 'model.json')
    config = read_json(run / 'config.json')
    # Ship the tested loading configuration, not its machine-specific symlinks.
    model_config = read_json(Path(model['load_view']) / 'config.json')
    tokenizer = Path(model['load_view']) / 'tokenizer.json'
    tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                       'generation_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt',
                       'tokenizer.model']
    mtp_fingerprints = None
    if trial['mtp']:
        from .quantization import tensor_fingerprints
        draft = Path(model['references']['mtp'])
        index = read_json(draft / 'model.safetensors.index.json')['weight_map']
        mtp_fingerprints = tensor_fingerprints(draft, [name for name in index if name.startswith('mtp.')])
    receipt = {
        'schema_version': 1, 'stage': 'portable_deployment',
        'experience': config.get('experience', {}),
        'trial': {key: trial[key] for key in ['id', 'label', 'attention', 'kernel', 'compile', 'graph', 'budget', 'mtp']},
        'identity': portable_identity(model['identity']), 'loading_config': model_config,
        'tokenizer_sha256': digest(tokenizer) if tokenizer.is_file() else None,
        'tokenizer_files_sha256': {name: digest(Path(model['load_view']) / name) for name in tokenizer_files
                                   if (Path(model['load_view']) / name).is_file()},
        'mtp_fingerprints': mtp_fingerprints, 'server': config['server'],
        'hardware': read_json(run / 'hardware.json'),
        'runtime_requirements': {'vllm': '0.33.0', 'torch': read_json(run / 'hardware.json')['torch'],
                                 'cuda': read_json(run / 'hardware.json')['cuda']},
        'default_model': model['path'],
        'default_mtp': config.get('handoff', {}).get('mtp_source'),
        'quality_scored': report['quality_scored'], 'weights_included': False,
        'tested_results': [row for row in report['rows'] if row['profile'] == trial['id']],
        'tested_source': read_json(run / 'source_manifest.json'),
        'health_path': '/health', 'cross_hardware_performance_validated': False,
    }
    write_json(destination / 'deployment.json', receipt)
    shutil.copy2(run / 'report.html', destination / 'report.html')
    script = '''#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${QUICKSTART_BIN:-quickstart}" deploy --bundle "$BUNDLE_DIR" "$@"
'''
    (destination / 'start.sh').write_text(script)
    (destination / 'start.sh').chmod(0o755)
    (destination / 'README.md').write_text(
        '# 可迁移部署包\n\n' + ('**快速体验产物，仅验证链路，尚不足以作为正式部署推荐。**\n\n'
                                    if config.get('experience', {}).get('smoke_only') else '') +
        '在公共镜像中挂载模型和本目录，执行：\n\n'
        '```bash\n./start.sh --model /models/chosen --mtp-model /models/mtp --host 0.0.0.0 --port 9001\n```\n\n'
        '未启用MTP的方案省略`--mtp-model`。`--smoke-test`执行一次完整请求后退出。'
        '代码方式使用`QUICKSTART_BIN=/path/to/project/quickstart ./start.sh ...`。'
        '权重SHA256、tokenizer与实际加载配置会检查；迁移无需保持原绝对路径。'
        '更换GPU架构须重新加速测试，本包不保证跨硬件仍最快。'
        '平台探针使用同一端口的`/health`；首次编译与预热期间需要启动宽限期。\n')
    return destination


def verify_input(bundle, model, mtp=None):
    receipt = read_json(Path(bundle) / 'deployment.json')
    if receipt.get('schema_version') != 1 or receipt.get('stage') != 'portable_deployment':
        raise ValueError('无效部署包')
    from .inference.planning import make_trial
    trial = receipt['trial']
    rebuilt = make_trial(**{name: trial[name] for name in ['attention', 'kernel', 'compile', 'graph', 'budget', 'mtp']})
    if rebuilt != trial or not receipt.get('tested_results') or any(
            row['profile'] != trial['id'] for row in receipt['tested_results']):
        raise ValueError('部署参数与实测方案身份不一致')
    identity, _ = inspect_model(model, payload_hashes=True)
    # Source config may be the exported original or the already-tested loading
    # view. We always restore the frozen loading config in a new private view.
    actual = portable_identity(identity)
    expected = receipt['identity']
    if actual['shards'] != expected['shards'] or actual['tensor_count'] != expected['tensor_count']:
        raise ValueError('模型权重与实测部署包不一致')
    if actual['config'] not in (expected['config'], receipt['loading_config']):
        raise ValueError('模型结构或量化配置与实测部署包不一致')
    if receipt.get('tokenizer_sha256') and digest(Path(model) / 'tokenizer.json') != receipt['tokenizer_sha256']:
        raise ValueError('Tokenizer已变化')
    for name, expected_hash in receipt.get('tokenizer_files_sha256', {}).items():
        if Path(name).name != name or digest(Path(model) / name) != expected_hash:
            raise ValueError('Tokenizer或聊天模板与实测不同: ' + name)
    if receipt['trial']['mtp']:
        if not mtp:
            raise ValueError('本方案启用了MTP，请指定 --mtp-model')
        from .quantization import tensor_fingerprints
        if tensor_fingerprints(mtp, receipt['mtp_fingerprints']) != receipt['mtp_fingerprints']:
            raise ValueError('MTP权重与实测部署包不一致')
    return receipt


def serve_bundle(root, bundle, config_path=None, model=None, mtp=None, smoke=False, host='127.0.0.1', port=9001, gpu=None):
    from .inference import assets, config as settings, planning
    from .inference.server import assert_idle
    from .workflow import resolve_config
    root, bundle = Path(root), Path(bundle)
    preliminary = read_json(bundle / 'deployment.json')
    model = Path(model or os.environ.get('MODEL_PATH') or preliminary['default_model']).expanduser().resolve()
    mtp = mtp or os.environ.get('MTP_MODEL_PATH') or preliminary.get('default_mtp')
    mtp = Path(mtp).expanduser().resolve() if mtp else None
    receipt = verify_input(bundle, model, mtp)
    config = settings.load(root, resolve_config(root, config_path, 'inference'))
    config['server'] = {**receipt['server'], 'port': port}
    if gpu is not None:
        config['runtime']['gpu'] = gpu
    directory = root / 'runs/deployments' / ('deploy-' + str(time.time_ns()))
    directory.mkdir(parents=True)
    from .inference.execution import snapshot
    snapshot(root, directory / 'source')
    with gpu_lease(root, config['runtime'].get('gpu', 0)):
        assert_idle(config, wait_seconds=40)
        hardware = assets.probe(root, config)
        if hardware['capability'] != receipt['hardware']['capability']:
            raise ValueError('GPU架构与实测不同，请在新硬件重新执行加速搜索')
        for key, expected in receipt['runtime_requirements'].items():
            if hardware.get(key) != expected:
                raise ValueError(f'部署运行时{key}与实测不符: {hardware.get(key)} != {expected}')
        prepared = assets.prepare(root, directory, config, model, mtp)
        write_json(Path(prepared['load_view']) / 'config.json', receipt['loading_config'])
        prepared['loading_files_sha256'][str(Path(prepared['load_view']) / 'config.json')] = digest(Path(prepared['load_view']) / 'config.json')
        write_json(directory / 'model.json', prepared)
        trial = receipt['trial']
        command = planning.server_command(config, prepared, trial, port=port, host=host)
        env = settings.environment(root, config, directory / 'source')
        env.update(STAGE2_REFERENCES=json.dumps(prepared['references']), STAGE2_MTP=str(trial['mtp']),
                   STAGE2_AUDIT_FILE=str(directory / 'audit.json'), VLLM_SERVER_DEV_MODE='0')
        write_json(directory / 'launch.json', {'command': command, 'trial': trial, 'hardware': hardware})
        base = 'http://127.0.0.1:' + str(port)
        with (directory / 'server.log').open('w') as log:
            process = subprocess.Popen(command, cwd=directory / 'source', env=env,
                                       start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
            try:
                started = time.monotonic()
                while True:
                    if process.poll() is not None:
                        raise RuntimeError('部署启动失败，查看 ' + str(directory / 'server.log'))
                    try:
                        endpoint(base, '/health', timeout=2)
                        break
                    except OSError:
                        if time.monotonic() - started > config['server']['startup_timeout_seconds']:
                            raise TimeoutError('部署启动超时')
                        time.sleep(1)
                if not read_json(directory / 'audit.json').get('passed'):
                    raise RuntimeError('部署加载审计失败')
                response = request_one(base, {'uid': 'deploy-smoke', 'group': 'smoke',
                    'messages': [{'role': 'user', 'content': '请用一句中文介绍你能提供什么帮助。'}]},
                    dict(config['benchmark'], max_tokens=128))
                write_json(directory / 'smoke.json', response)
                if response['status'] != 'success':
                    raise RuntimeError('部署真实请求未完整结束')
                print(json.dumps({'status': 'serving_validated', 'url': base + '/v1/chat/completions',
                                  'answer': response['content'], 'smoke_only': smoke}, ensure_ascii=False), flush=True)
                if not smoke:
                    process.wait()
                    raise RuntimeError('服务意外停止')
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
                write_json(directory / 'cleanup.json', {'exit_code': process.returncode})
    return directory


def main(argv=None, install_signals=True):
    parser = argparse.ArgumentParser(description='启动已验证的可迁移部署包')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--mtp-model', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9001)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args(argv)
    if install_signals:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    serve_bundle(args.root.resolve(), args.bundle.resolve(), args.config, args.model, args.mtp_model,
                 args.smoke_test, args.host, args.port, args.gpu)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print('DEPLOYMENT_ERROR: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
