"""Serve/export only a fully retested, frozen stage-two candidate."""
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from pipeline.benchmark import endpoint, request_one
from pipeline.common import digest, read_json, write_json
from .assets import verify_model
from .config import environment
from .execution import gpu_lock
from .report import generate
from .server import assert_idle


def select(run, profile_id, cache=None):
    report = generate(run)
    if profile_id == 'recommended':
        profile_id = report['best_by_cache'].get(cache) if cache else report['recommendation']
    if (not profile_id or profile_id not in {row['profile'] for row in report['rows']}
            or not report['quality_gates'].get(profile_id) or not report['quality_comparison_ready']):
        raise ValueError('只能选择完成全部复测的候选，不能部署筛选阶段或失败组合')
    trial = read_json(run / 'candidates.json')[profile_id]
    write_json(run / 'selected.json', {'profile': trial, 'cache': cache, 'selected_at': time.time(),
                                       'quality_scored': report['quality_scored']})
    return export(run, profile_id)


def export(run, profile_id):
    config = read_json(run / 'config.json')
    model = read_json(run / 'model.json')
    trial = read_json(run / 'candidates.json')[profile_id]
    report = read_json(run / 'report.json')
    if (profile_id not in {row['profile'] for row in report['rows']} or not report['quality_gates'].get(profile_id)
            or not report['quality_comparison_ready']):
        raise ValueError('候选未通过完整复测')
    directory = run / 'deployment' / profile_id
    directory.mkdir(parents=True, exist_ok=True)
    env = environment(Path(config['root']), config, run / 'source')
    allowed = {'PATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_VISIBLE_DEVICES', 'PYTHONPATH',
               'MAX_JOBS', 'FLASHINFER_NVCC_THREADS', 'ACCEL60_DRAFT_CONFIG_FIX', 'VLLM_CACHE_ROOT', 'VLLM_RPC_BASE_PATH',
               'FLASHINFER_WORKSPACE_BASE', 'CUDA_CACHE_PATH', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR'}
    command = [str(Path(config['root']) / 'accelerate'), 'serve', '--run-id', run.name, '--profile', profile_id]
    write_json(directory / 'deployment.json', {'stage': 2, 'profile': trial, 'model': model,
               'hardware': read_json(run / 'hardware.json'), 'image': config['runtime'].get('image'),
               'environment': {key: env[key] for key in allowed if key in env},
               'source_manifest': read_json(run / 'source_manifest.json'), 'start_command': command,
               'health_path': '/health', 'default_port': config['server']['port'],
               'weights_included': False, 'scope': 'tested_model_environment_and_candidate'})
    (directory / 'start.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec ' + shlex.join(command) + ' "$@"\n')
    (directory / 'start.sh').chmod(0o755)
    (directory / 'README.md').write_text(
        '# 第二阶段已测部署方案\n\n完整参数、模型/MTP路径、权重SHA256、GPU/依赖版本与启动命令见`deployment.json`。'
        '权重和容器镜像未包含；迁移时需准备匹配环境、全部模型加载视图、快照代码和挂载路径。\n\n'
        '当前环境启动：\n\n```bash\n' + shlex.quote(str(directory / 'start.sh')) + '\n```\n\n'
        '一次真实回复后停止：在命令后添加`--smoke-test`。指定外部端口：添加`--host 0.0.0.0 --port 9001`。'
        '平台探针使用相同端口的`/health`，宽限时间需覆盖实际编译和模型加载。\n'
    )
    return directory


def serve(root, run, profile_id=None, host='127.0.0.1', port=None, smoke=False):
    config = read_json(run / 'config.json')
    model = read_json(run / 'model.json')
    if profile_id is None:
        profile_id = read_json(run / 'selected.json')['profile']['id']
    report = read_json(run / 'report.json')
    if (profile_id not in {row['profile'] for row in report['rows']} or not report['quality_gates'].get(profile_id)
            or not report['quality_comparison_ready']):
        raise ValueError('不能启动未完成完整复测的候选')
    trial = read_json(run / 'candidates.json')[profile_id]
    verify_model(model)
    directory = run / 'deployments' / str(time.time_ns())
    directory.mkdir(parents=True)
    command = list(trial['command'])
    command[command.index('--host') + 1] = host
    command[command.index('--port') + 1] = str(port or config['server']['port'])
    env = environment(root, config, run / 'source')
    env.update(STAGE2_REFERENCES=__import__('json').dumps(model['references']),
               STAGE2_MTP=str(trial['mtp']), STAGE2_AUDIT_FILE=str(directory / 'audit.json'),
               VLLM_SERVER_DEV_MODE='0')
    settings = dict(config, server=dict(config['server'], port=port or config['server']['port']))
    base = 'http://127.0.0.1:' + str(settings['server']['port'])
    with gpu_lock(root, gpu=config['runtime'].get('gpu', 0)):
        assert_idle(settings, wait_seconds=1)
        write_json(directory / 'launch.json', {'profile': trial, 'command': command})
        with (directory / 'server.log').open('w') as log:
            process = subprocess.Popen(command, cwd=run / 'source', env=env, start_new_session=True,
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                started = time.monotonic()
                while True:
                    if process.poll() is not None:
                        raise RuntimeError('部署进程退出，查看 ' + str(directory / 'server.log'))
                    try:
                        endpoint(base, '/health', timeout=2)
                        break
                    except OSError:
                        if time.monotonic() - started > config['server']['startup_timeout_seconds']:
                            raise TimeoutError('部署启动超时')
                        time.sleep(1)
                response = request_one(base, {'uid': 'deployment', 'group': 'smoke',
                    'messages': [{'role': 'user', 'content': '请用一句中文介绍你能提供什么帮助。'}]},
                    dict(config['benchmark'], max_tokens=128))
                write_json(directory / 'smoke.json', response)
                if response['status'] != 'success':
                    raise RuntimeError('部署回复未完整结束')
                print(__import__('json').dumps({'status': 'serving_validated', 'url': base,
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
