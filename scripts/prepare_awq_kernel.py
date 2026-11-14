"""为历史 GPTQModel 产物预编译 SM80 AWQ 内核；正式入口只校验复用，不安装依赖或重新编译。

--build 可在无 GPU 的 CPU 容器中使用 nvcc 编译；不会读取模型权重。
--require-gpu 在新单卡任务内执行小矩阵自检，再交给原量化流程。
使用固定 GPTQModel 7.5.0 的 JIT 指纹和缓存目录，保持共享依赖只读。
"""
import argparse
import fcntl
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

EXPECTED_ARCH = '8.0'
EXPECTED_GPTQMODEL = '7.5.0'
MANIFEST_NAME = 'awq_kernel_ready.json'


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_environment(environment):
    if environment.get('TORCH_CUDA_ARCH_LIST') != EXPECTED_ARCH:
        raise ValueError('此预编译工具要求 TORCH_CUDA_ARCH_LIST=8.0（SM80）')
    for key in ('GPTQMODEL_KERNEL_REBUILD', 'GPTQMODEL_AWQ_FORCE_REBUILD'):
        if environment.get(key, '').lower() not in ('', '0', 'false', 'no', 'off'):
            raise ValueError(f'{key} 会清除已验证内核，请关闭强制重编译')
    if not environment.get('GPTQMODEL_AWQ_BUILD_ROOT'):
        raise ValueError('必须指定独立 GPTQMODEL_AWQ_BUILD_ROOT')


def verify_manifest(manifest, expected, library):
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'AWQ 内核环境或源码指纹不匹配：{key}；请先离线重新预编译')
    if not library.is_file() or manifest.get('library_sha256') != file_sha256(library):
        raise ValueError('AWQ 预编译文件缺失或 SHA256 不符；禁止自动重新编译')


def gpu_smoke(torch, extension):
    """在独立单卡 CUDA 环境执行；验证两种激活精度的反量化和 GEMM。"""
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('AWQ 单卡预检需要恰好一张可见 CUDA GPU')
    if torch.cuda.get_device_capability(0) != (8, 0):
        raise ValueError('此预编译内核仅支持 SM80，当前 GPU 计算能力不匹配')
    checks = []
    # 每个 nibble 的 q=3、zero=1、scale=1/8，因此真实权重恒为 1/4。
    packed = torch.full((128, 16), 0x33333333, dtype=torch.int32, device='cuda')
    zeros = torch.full((1, 16), 0x11111111, dtype=torch.int32, device='cuda')
    for dtype in (torch.float16, torch.bfloat16):
        scales = torch.full((1, 128), 0.125, dtype=dtype, device='cuda')
        weights = extension.op('dequantize_weights')(packed, scales, zeros, 0, 0, 0, False)
        torch.testing.assert_close(weights, torch.full_like(weights, 0.25), rtol=0, atol=0)
        for rows in (1, 16):
            inputs = torch.ones((rows, 128), dtype=dtype, device='cuda')
            actual = extension.op('gemm_forward')(inputs, packed, scales, zeros, 1, False)
            torch.testing.assert_close(actual, torch.full_like(actual, 32), rtol=0, atol=0)
        checks.append(str(dtype))
    torch.cuda.synchronize()
    return {'device': torch.cuda.get_device_name(0), 'capability': [8, 0],
            'dequantize_and_gemm_passed': checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', help='仅准备阶段允许编译；正式入口不用此选项')
    parser.add_argument('--require-gpu', action='store_true', help='新任务获卡后先验证 GPU 内核')
    args = parser.parse_args(argv)
    check_environment(os.environ)
    if version('gptqmodel') != EXPECTED_GPTQMODEL:
        raise ValueError('AWQ 预编译只支持已冻结的 GPTQModel 7.5.0')
    import torch
    from gptqmodel.utils.awq import _AWQ_TORCH_OPS_EXTENSION as extension

    base = Path(os.environ['GPTQMODEL_AWQ_BUILD_ROOT']).resolve()
    if args.build:
        base.mkdir(parents=True, exist_ok=True)
    manifest_path = base / MANIFEST_NAME
    build_root = extension.build_root()
    library = build_root / (extension.name + '.so')
    expected = {'version': 'awq-sm80-prebuilt-v1', 'arch': EXPECTED_ARCH,
                'gptqmodel': EXPECTED_GPTQMODEL, 'torch': torch.__version__,
                'torch_cuda': torch.version.cuda, 'cxx11_abi': torch._C._GLIBCXX_USE_CXX11_ABI,
                'fingerprint': build_root.name, 'library': str(library),
                'required_ops': list(extension.required_ops)}
    started = time.monotonic()
    if args.build:
        # 单独锁住此 AWQ 编译目录，不接触正在运行的其他版本的锁或缓存。
        with (base / '.prepare.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not manifest_path.exists():
                from torch.utils.cpp_extension import load
                build_root.mkdir(parents=True, exist_ok=True)
                load(name=extension.name, sources=extension._resolve_sequence(extension.sources),
                     build_directory=str(build_root), is_python_module=False, verbose=True,
                     extra_cflags=extension._resolved_extra_cflags(),
                     extra_cuda_cflags=extension._resolved_extra_cuda_cflags(),
                     extra_include_paths=extension._resolved_extra_include_paths(),
                     extra_ldflags=extension._resolve_sequence(extension.extra_ldflags))
                if not extension.load():
                    raise RuntimeError(extension.last_error_message())
                manifest = {**expected, 'library_sha256': file_sha256(library),
                            'build_elapsed_seconds': round(time.monotonic() - started, 2),
                            'gpu_validated': False}
                temporary = manifest_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
                temporary.replace(manifest_path)
    if not manifest_path.is_file():
        raise ValueError('AWQ 内核尚未预编译；请先在 CPU 环境执行 --build')
    manifest = json.loads(manifest_path.read_text())
    verify_manifest(manifest, expected, library)
    # 显式加载，失败直接停止；绝不回退到 CUDA JIT 或其他量化算法。
    if not extension._ops_available():
        torch.ops.load_library(str(library))
    if not extension.load():
        raise RuntimeError(extension.last_error_message())
    result = {'status': 'awq_kernel_verified', **expected,
              'library_sha256': manifest['library_sha256'], 'model_requests': 0,
              'gpu_validated': False}
    if args.require_gpu:
        result['gpu_smoke'] = gpu_smoke(torch, extension)
        result['gpu_validated'] = True
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
