"""Real device check; distinct from successful model quantization or serving."""
import json
import sys
import torch

report = {'event': 'inferboost_cuda_check', 'passed': False,
          'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'model_inference': 'not_tested'}
try:
    if not torch.cuda.is_available():
        raise RuntimeError('容器无法访问CUDA；检查GPU挂载、NVIDIA Container Toolkit及驱动。')
    devices = []
    for index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(index)
        if capability < (8, 0):
            raise RuntimeError(f'GPU{index}能力SM{capability[0]}{capability[1]}低于SM80；本镜像不支持V100。')
        with torch.cuda.device(index):
            values = torch.ones((256, 256), dtype=torch.bfloat16, device='cuda')
            product = values @ values
            torch.cuda.synchronize()
            if product.float().mean().item() != 256:
                raise RuntimeError('CUDA矩阵计算校验失败')
        devices.append({'index': index, 'name': torch.cuda.get_device_name(index), 'capability': capability})
    report.update(passed=True, devices=devices)
except Exception as error:
    report['error'] = str(error)
print(json.dumps(report, ensure_ascii=False), flush=True)
raise SystemExit(0 if report['passed'] else 1)
