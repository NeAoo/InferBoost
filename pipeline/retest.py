"""Branch measured candidates into fresh acceleration runs; source runs stay read-only."""
import copy
from pathlib import Path

from .common import identifier, read_json


def source(directory):
    directory = Path(directory).expanduser().resolve()
    if directory.parent.name != 'workflows' or directory.parent.parent.name != 'runs':
        raise ValueError('来源应为工作目录下的 runs/workflows/<任务名>')
    from .jobs import status
    root = directory.parents[2]
    if status(root, directory.name)['alive']:
        raise ValueError('来源任务仍在运行，请先等待完成或停止任务')
    request = read_json(directory / 'request.json')
    lineage = request.get('retest_source')
    quant = (Path(lineage['quantization_directory']) if lineage else
             root / 'runs' / identifier(request['quantization_run']))
    if read_json(quant / 'status.json').get('status') not in ('completed', 'completed_with_failures'):
        raise ValueError('来源量化与评测尚未完成，不能重新选择模型')
    return directory, quant, request


def configure(args):
    directory, quant, request = source(args.from_workflow)
    if not args.candidate or args.candidate == 'recommended':
        raise ValueError('重新选择需要明确的 --candidate，例如 awq4 或 gptq4')
    candidate = identifier(args.candidate)
    if read_json(quant / 'candidates' / candidate / 'status.json').get('status') != 'completed':
        raise ValueError('只能选择已完成评测的候选')
    # Inspect metadata here. Full weight and loading audits run in the worker.
    receipt = read_json(quant / 'candidates' / candidate / 'model_manifest.json')
    args.model = Path(receipt['load_view'])
    if not (args.model / 'config.json').is_file():
        raise ValueError('候选模型不可访问，请挂载原工作目录及模型目录')
    quant_config = read_json(quant / 'config.json')
    mtp = quant_config.get('mtp_source') or request.get('mtp')
    args.mtp_model = Path(mtp) if mtp else None
    args.skip_quantization = True
    return directory, quant


def configs(args):
    directory, quant, request = source(args.from_workflow)
    inference = copy.deepcopy(read_json(directory / 'inference.json'))
    old_accel = directory.parents[2] / 'runs/inference' / identifier(request['inference_run'])
    # Use the exact measured cohorts, even if the original dataset was later edited.
    for section, name in [('benchmark', 'inputs.json'), ('quality', 'quality_inputs.json')]:
        snapshot = old_accel / name
        if snapshot.is_file():
            inference[section]['dataset'] = str(snapshot)
            inference[section]['sample_limit'] = None
    inference.pop('handoff', None)
    inference['retest_source'] = {'workflow': str(directory), 'quantization_directory': str(quant),
                                'candidate': args.candidate}
    return read_json(quant / 'config.json'), inference


def selection(args):
    from .quantization import select
    _, quant, _ = source(args.from_workflow)
    return select(quant.parents[1], quant.name, args.candidate, save=False)
