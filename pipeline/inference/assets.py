"""Read-only checkpoint inspection and private loading views."""
import copy
import json
from pathlib import Path
import shutil
import struct
import subprocess

from pipeline.common import capture, digest, write_json
from pipeline.models import inspect_model
from .config import environment


def inspect(path):
    path = Path(path).resolve()
    identity, tensors = inspect_model(path)
    config = identity['config']
    quant = config.get('quantization_config') or {}
    groups = list((quant.get('config_groups') or {}).values())
    bits = quant.get('bits') or quant.get('weight_bits')
    if bits is None:
        values = {group.get('weights', {}).get('num_bits') for group in groups if group.get('weights')}
        bits = next(iter(values)) if len(values) == 1 else None
    activations = [group.get('input_activations') for group in groups if group.get('input_activations')]
    activation_bits = min([item.get('num_bits', 16) for item in activations] or [16])
    activation_type = 'float' if any(item.get('type') == 'float' for item in activations) else 'int'
    method = quant.get('quant_method', 'none').replace('_', '-')
    text = config.get('text_config', config)
    mtp = {key: value for key, value in tensors.items() if key.startswith('mtp.')}
    bf16_mtp = bool(mtp) and 'mtp.fc.weight' in mtp and all(value['dtype'] == 'BF16' for value in mtp.values())
    return {
        'path': str(path), 'identity': identity, 'method': method, 'bits': bits,
        'activation_bits': activation_bits, 'activation_type': activation_type,
        'precision': f'W{bits}A{activation_bits}' if bits else ('BF16/FP16' if not quant else 'checkpoint_defined'),
        'model_type': config.get('model_type'), 'text_model_type': text.get('model_type'),
        'text_config': text, 'bf16_mtp_present': bf16_mtp,
        'mtp_tensors': len(mtp), 'tensor_headers': tensors,
    }


def probe(root, config):
    code = '''import importlib.util,json,platform,torch
from importlib.metadata import version
print(json.dumps({"gpu":torch.cuda.get_device_name(),"capability":torch.cuda.get_device_capability(),
"torch":torch.__version__,"cuda":torch.version.cuda,"python":platform.python_version(),
"vllm":version("vllm"),"flashinfer":bool(importlib.util.find_spec("flashinfer")),
"humming":bool(importlib.util.find_spec("humming")),
"pandas":version("pandas")}))'''
    result = capture([config['runtime']['python'], '-c', code], env=environment(root, config), timeout=120)
    hardware = json.loads(result.splitlines()[-1])
    if hardware['vllm'] != '0.33.0':
        raise ValueError('当前第二阶段适配器要求 vLLM 0.33.0；其他版本须单独验收')
    hardware['uuid'] = capture(['nvidia-smi', '-i', str(config['runtime'].get('gpu', 0)),
                                '--query-gpu=uuid', '--format=csv,noheader'])
    hardware['driver'] = capture(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']).splitlines()[0]
    return hardware


def inspect_mtp(path):
    """MTP loading views may index only selected tensors from full source shards."""
    path = Path(path).resolve()
    index = path / 'model.safetensors.index.json'
    if not index.is_file():
        return inspect(path)
    config = json.loads((path / 'config.json').read_text())
    mapping = json.loads(index.read_text())['weight_map']
    headers = {}
    for name in set(mapping.values()):
        if Path(name).name != name:
            raise ValueError('MTP分片路径非法')
        with (path / name).open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            if not 2 <= length <= 64 * 1024**2:
                raise ValueError('MTP分片文件头异常')
            header = json.loads(stream.read(length))
        for key, value in header.items():
            if key != '__metadata__' and mapping.get(key) == name:
                headers[key] = value
    mtp = {key: value for key, value in headers.items() if key.startswith('mtp.')}
    if set(headers) != set(mapping):
        raise ValueError('MTP索引声明的张量不完整')
    return {'path': str(path), 'text_config': config.get('text_config', config),
            'bf16_mtp_present': 'mtp.fc.weight' in mtp and all(item['dtype'] == 'BF16' for item in mtp.values())}


def verify_loading_files(model):
    for name, expected in model['loading_files_sha256'].items():
        if digest(name) != expected:
            raise ValueError('已测加载配置或审计参考已变化: ' + name)


def verify_model(model):
    identity, _ = inspect_model(model['path'], payload_hashes=True)
    if identity != model['identity']:
        raise ValueError('模型权重已变化，不能复用原推理加速结果')
    verify_loading_files(model)


def prepare(root, run_dir, config, model_path, mtp_path=None):
    model = inspect(model_path)
    identity, tensors = inspect_model(model['path'], payload_hashes=True)
    view = run_dir / 'model'
    view.mkdir()
    for path in Path(model['path']).iterdir():
        if path.is_file() and path.suffix in ('.json', '.jinja', '.txt', '.model'):
            shutil.copy2(path, view / path.name)
    for name in identity['shards']:
        (view / name).symlink_to((Path(model['path']) / name).resolve())
    settings = copy.deepcopy(identity['config'])
    quant = settings.get('quantization_config')
    qwen35 = str(model['model_type']).startswith('qwen3_5')
    protected = []
    if qwen35:
        skip_key = 'ignore' if model['method'] == 'compressed-tensors' else 'modules_to_not_convert'
        skipped = set((quant or {}).get(skip_key) or [])
        for name, tensor in tensors.items():
            if name.endswith(('.linear_attn.in_proj_a.weight', '.linear_attn.in_proj_b.weight',
                              '.linear_attn.in_proj_ba.weight')):
                if tensor['dtype'] not in ('BF16', 'F16'):
                    raise ValueError('未识别的高精度保护投影: ' + name)
                module = name[:-7]
                if skip_key == 'modules_to_not_convert':
                    module = module[module.index('layers.'):]
                skipped.add(module)
                if module.endswith(('.in_proj_a', '.in_proj_b')):
                    skipped.add(module.rsplit('.', 1)[0] + '.in_proj_ba')
                protected.append(module)
        if quant:
            quant[skip_key] = sorted(skipped)
            write_json(view / 'config.json', settings)
    references = {}
    source = inspect_mtp(mtp_path) if mtp_path else model
    mtp_supported = qwen35 and source['bf16_mtp_present']
    if mtp_path and not mtp_supported:
        raise ValueError('当前MTP适配器要求匹配Qwen3.5且草稿张量保持BF16')
    if mtp_supported:
        for key in ['hidden_size', 'vocab_size', 'num_hidden_layers']:
            if source['text_config'].get(key) != model['text_config'].get(key):
                raise ValueError('MTP与主模型结构不一致: ' + key)
    helper = run_dir / 'source/scripts/prepare_qwen_assets.py'
    if qwen35 and protected:
        command = [config['runtime']['python'], str(helper), '--model', str(view), '--output', str(run_dir / 'assets')]
        if mtp_supported:
            command += ['--mtp-source', source['path']]
        else:
            command += ['--audit-only']
        try:
            subprocess.run(command, env=environment(root, config, run_dir / 'source'), check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError('MTP或保护权重准备失败: ' + error.stderr[-2000:]) from error
        references['protected'] = str(run_dir / 'assets/expected_bf16.json')
        if mtp_supported:
            references.update(mtp=str(run_dir / 'assets/draft'), mtp_reference=str(run_dir / 'assets/mtp_reference.json'))
    elif mtp_supported:
        mtp_supported = False
    receipt = {key: value for key, value in model.items() if key != 'tensor_headers'}
    loading_files = [path for path in view.iterdir() if path.suffix in ('.json', '.jinja', '.txt', '.model')]
    loading_files.extend((run_dir / 'assets').rglob('*.json'))
    receipt.update(identity=identity, load_view=str(view), references=references,
                   mtp_supported=mtp_supported, mtp_policy='separate_bf16_draft',
                   protected_modules=protected, source_config_sha256=digest(Path(model['path']) / 'config.json'),
                   loading_files_sha256={str(path): digest(path) for path in loading_files})
    write_json(run_dir / 'model.json', receipt)
    return receipt
