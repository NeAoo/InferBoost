"""Export the tested deployment recipe without copying weights or credentials."""
from pathlib import Path
import shlex

from .common import capture, identifier, inside, read_json, runtime_environment, source_digest, write_json
from .runtime import build_command


ENVIRONMENT_KEYS = (
    'CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'LD_LIBRARY_PATH', 'PATH', 'PYTHONPATH',
    'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'TOKENIZERS_PARALLELISM',
    'PYTHONDONTWRITEBYTECODE', 'PYTHONUNBUFFERED', 'TMPDIR', 'XDG_CACHE_HOME',
    'HF_HOME', 'CUDA_CACHE_PATH', 'VLLM_CACHE_ROOT', 'VLLM_RPC_BASE_PATH', 'TRITON_CACHE_DIR',
    'TORCHINDUCTOR_CACHE_DIR', 'TORCH_EXTENSIONS_DIR', 'FLASHINFER_WORKSPACE_BASE',
    'MAX_JOBS', 'FLASHINFER_NVCC_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS',
    'ACCEL60_DRAFT_CONFIG_FIX',
)


def tested_profile(run_dir, profile_id):
    identifier(profile_id)
    config = read_json(run_dir / ('execution_config.json' if (run_dir / 'execution_config.json').exists() else 'config.json'))
    profiles = [profile for profile in config['profiles'] if profile['id'] == profile_id]
    directory = run_dir / 'profiles' / profile_id
    if not profiles or read_json(directory / 'status.json')['status'] != 'completed':
        raise ValueError('只能部署本轮已完成实测的方案')
    launch = read_json(directory / 'launch.json')
    if launch.get('profile') != profiles[0]:
        raise ValueError('方案参数与实测启动记录不一致；请建立新批次重新测试')
    model = read_json(run_dir / 'model_manifest.json')
    command = build_command(config, profiles[0], model['load_view'], config['server']['port'])
    if command != launch['command']:
        raise ValueError('服务参数或启动实现与实测命令不一致；请复测后部署')
    return config, profiles[0], launch


def export(root, run_dir, profile_id):
    config, profile, tested = tested_profile(run_dir, profile_id)
    model = read_json(run_dir / 'model_manifest.json')
    if not all((Path(model[key]) / 'config.json').is_file() for key in ['source_model', 'load_view']):
        raise ValueError('当前环境无法访问实测模型和加载视图；报告副本不能直接作为可运行部署环境')
    if not Path(config['runtime']['python']).is_file():
        raise ValueError('当前环境无法访问实测推理Python，不能导出可运行配方')
    directory = inside(root, run_dir / 'deployment-package' / profile_id)
    directory.mkdir(parents=True, exist_ok=True)
    environment = runtime_environment(root, config)
    environment = {key: environment[key] for key in ENVIRONMENT_KEYS if key in environment}
    environment['VLLM_SERVER_DEV_MODE'] = '0'
    audit = config.get('audit', {})
    if audit.get('expected_bf16'):
        environment['EXPECTED_BF16_FILE'] = audit['expected_bf16']
        environment['LOADING_AUDIT_FILE'] = str(directory / 'loading_audit.json')
    if profile.get('mtp') and audit.get('mtp_reference'):
        environment['MTP_AUDIT_REFERENCE'] = audit['mtp_reference']
    else:
        # The recipe explicitly clears this variable for non-MTP deployments.
        environment.pop('MTP_AUDIT_REFERENCE', None)
    audit_file = run_dir / 'profiles' / profile_id / 'loading_audit.json'
    loading = read_json(audit_file) if audit_file.exists() else {}
    doctor = read_json(run_dir / 'doctor.json')
    python_version = capture([config['runtime']['python'], '-c', 'import platform; print(platform.python_version())'], timeout=20)
    port = config['server']['port']
    command = build_command(config, profile, model['load_view'], port)
    if command != tested['command']:
        raise ValueError('当前启动实现与实测命令不一致；请复测后导出')
    start = [str(root / 'quickstart'), 'serve', '--run-id', run_dir.name, '--profile', profile_id]
    receipt = {
        'schema_version': 1, 'scope': 'tested_current_environment',
        'run_id': run_dir.name, 'profile': profile, 'server': config['server'],
        'model': model, 'mtp_model': config.get('mtp_model') if profile.get('mtp') else None,
        'audit_references': audit, 'versions': loading.get('versions') or {
            key: doctor.get(key) for key in ['vllm', 'torch', 'cuda', 'flashinfer']},
        'gpu': {'name': doctor['gpu'], 'capability': doctor.get('capability')},
        'runtime_build': {'python': python_version, 'python_executable': config['runtime']['python'],
                          'torch': doctor.get('torch'), 'torch_cuda': doctor.get('cuda')},
        'pipeline_code_sha256': source_digest(root),
        'image': config['runtime'].get('image'), 'image_validated': False,
        'serving_command': command, 'environment': environment,
        'start_command': start, 'health_url': f'http://127.0.0.1:{port}/health',
        'chat_url': f'http://127.0.0.1:{port}/v1/chat/completions',
        'startup_timeout_seconds': config['server'].get('startup_timeout_seconds', 1800),
        'quality_validated': False, 'weights_included': False,
        'migration_requires': ['匹配的镜像或Python/CUDA环境', '完整的主模型加载视图及权重',
                              '启用MTP时的草稿加载视图及权重', '项目代码、审计引用文件与路径映射'],
    }
    write_json(directory / 'deployment.json', receipt)
    (directory / 'start.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec ' + shlex.join(start) + ' "$@"\n')
    (directory / 'start.sh').chmod(0o755)
    (directory / 'README.md').write_text(
        '# 已验证方案的部署资料\n\n'
        f'来源：`{run_dir.name}`；方案：`{profile_id}`。\n\n'
        '这是当前已核验环境的部署配方。模型权重与镜像未包含在此目录中；迁移到另一台机器时，'
        '须准备匹配环境、模型/MTP加载视图、项目代码及挂载路径。不能将此目录当作独立容器镜像。\n\n'
        '`deployment.json`保存模型清单与SHA256、环境版本、GPU、参数、启动命令、非敏感环境变量及健康检查地址。\n\n'
        '包含MTP修复开关时，需使用配套的已修复运行环境；仅安装相同版本号的原版框架不能代替该环境。\n\n'
        '在当前环境启动：\n\n```bash\n' + shlex.quote(str(directory / 'start.sh')) + '\n```\n\n'
        '仅验证启动与一次回复，随后停止：\n\n```bash\n' + shlex.quote(str(directory / 'start.sh')) + ' --smoke-test\n```\n\n'
        '指定对外端口：\n\n```bash\n' + shlex.quote(str(directory / 'start.sh')) + ' --host 0.0.0.0 --port 9001\n```\n\n'
        '平台探针使用同一对外端口的`/health`；启动宽限时间需覆盖实际加载、编译与预热。'
        '对外认证、TLS和流量入口由部署平台配置。\n\n'
        '修改内核或调度预算后，请新建测试批次；通过后重新选择并导出。启动脚本固定方案ID，'
        '不会因为后来选择了另一方案而静默切换。\n'
    )
    return directory
