"""AWQ 原始激活缓存的精确去重，仅作用于当前量化子进程。

原 recorder 仍先执行：padding、dtype、顺序及每次捕获均保持原样。
仅将同一专家、同一批次、同一次调用且逐字节相等的 gate/up CPU 缓存
共享存储；不改 256 条输入、token-row 预算/选取、统计或量化搜索代码。
下游原聚合器分别生成独立工作张量，禁止共享张量进入原地缩放。
"""
import hashlib
import inspect
import json
import re
import threading
import time
import weakref
from contextlib import contextmanager
from pathlib import Path

PATCH_VERSION = 'awq-exact-gate-up-cache-dedup-v1'
PROCESSOR_SHA256 = '51be5c05b800fa07cfd6506646575280762762e2273ab0016ae460ad9528b42c'
GIB = 1024 ** 3
PAIR_PATTERN = re.compile(r'^(.*\bexperts\.\d+)\.(gate_proj|up_proj)$')


def optimization_metadata():
    return {'version': PATCH_VERSION, 'processor_sha256': PROCESSOR_SHA256,
            'comparison': 'exact-cpu-bytes', 'sampling_changed': False,
            'calibration_changed': False, 'scope': 'routed-expert-gate-up-only'}


def verify_processor():
    from importlib.metadata import version
    from gptqmodel.looper.awq_processor import AWQProcessor
    source = Path(inspect.getsourcefile(AWQProcessor))
    if version('gptqmodel') != '7.5.0' or hashlib.sha256(source.read_bytes()).hexdigest() != PROCESSOR_SHA256:
        raise ValueError('AWQ 缓存去重依赖指纹不符，已在加载权重前停止')
    return AWQProcessor


def counterpart(name):
    match = PAIR_PATTERN.fullmatch(name)
    if match is None:
        return None
    return match[1] + ('.up_proj' if match[2] == 'gate_proj' else '.gate_proj')


def same_bytes(left, right):
    """包括 NaN payload、正负零也逐字节核对；不产生大型连续副本。"""
    import torch
    if (left.device.type != 'cpu' or right.device.type != 'cpu'
            or left.dtype != right.dtype or left.shape != right.shape
            or not left.is_contiguous() or not right.is_contiguous()):
        return False
    return torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8))


def retained_storage_bytes(tasks):
    """仅用于验证/每层审计，不持有缓存引用。"""
    storages = {}
    logical = 0
    for entry in tasks.values():
        for tensor in entry.get('inputs', []):
            logical += tensor.numel() * tensor.element_size()
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return {'logical_bytes': logical, 'unique_storage_bytes': sum(storages.values())}


@contextmanager
def exact_deduplicated_capture(report_path=None):
    """退出时恢复类方法；共享安装文件不写入，GPTQ 类完全不改。"""
    processor_class = verify_processor()
    original_record = processor_class._record_input_feature
    original_features = processor_class._layer_input_features
    if getattr(original_record, '_exact_cache_dedup', False):
        raise RuntimeError('AWQ 缓存去重不能重复安装')
    states = weakref.WeakKeyDictionary()
    states_lock = threading.Lock()
    totals = {'captures_compared': 0, 'captures_shared': 0,
              'unequal_captures': 0, 'cumulative_duplicate_bytes_removed': 0,
              'layers_aggregated': 0}

    def state_for(processor):
        with states_lock:
            if processor not in states:
                states[processor] = {'lock': threading.RLock()}
            return states[processor]

    def record(processor, name, feature, *, dedupe_batch=False):
        other_name = counterpart(name)
        policy = processor._feature_aggregation_policy(name) if other_name else None
        if dedupe_batch or not policy or policy.get('mode') != 'token_rows':
            return original_record(processor, name, feature, dedupe_batch=dedupe_batch)
        # 不依赖 GPU 指针或推断两个输入相等。先让原 recorder 完成传输。
        with state_for(processor)['lock']:
            original_record(processor, name, feature, dedupe_batch=False)
            with processor.lock:
                entry = processor.tasks[name]
                position = len(entry['inputs']) - 1
                batch_index = entry['batch_indices'][position]
                if batch_index is None or batch_index < 0:
                    return
                positions = entry.setdefault('_exact_dedup_positions', {}).setdefault(batch_index, [])
                occurrence = len(positions)
                positions.append(position)
                other = processor.tasks.get(other_name) or {}
                other_positions = other.get('_exact_dedup_positions', {}).get(batch_index, [])
                if occurrence >= len(other_positions):
                    return
                other_position = other_positions[occurrence]
                if other_position >= len(other.get('inputs', [])) or other['batch_indices'][other_position] != batch_index:
                    raise RuntimeError('AWQ 去重缓存生命周期异常，拒绝复用旧批次')
                current, previous = entry['inputs'][position], other['inputs'][other_position]
                totals['captures_compared'] += 1
                if same_bytes(current, previous):
                    if current.untyped_storage().data_ptr() != previous.untyped_storage().data_ptr():
                        totals['cumulative_duplicate_bytes_removed'] += current.untyped_storage().nbytes()
                    entry['inputs'][position] = previous
                    totals['captures_shared'] += 1
                else:
                    # 数据不同则原样保留两份，不能为了省内存改变校准输入。
                    totals['unequal_captures'] += 1

    def features(processor, layer_state):
        with state_for(processor)['lock']:
            storage_before = retained_storage_bytes(processor.tasks)
            result = original_features(processor, layer_state)
            # apply_scale 会原地 div_；不能只比较数值而忽略存储别名。
            for name, tensor in result.items():
                other = result.get(counterpart(name))
                if (other is not None and tensor.numel() and other.numel()
                        and tensor.untyped_storage().data_ptr() == other.untyped_storage().data_ptr()):
                    raise RuntimeError('AWQ 聚合后 gate/up 仍共享存储，已在缩放前停止')
            for entry in processor.tasks.values():
                entry.pop('_exact_dedup_positions', None)
            totals['layers_aggregated'] += 1
            if report_path is not None:
                with Path(report_path).open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'event': 'layer_cache_aggregated', 'timestamp': time.time(),
                        'layer_index': getattr(layer_state, 'layer_index', None),
                        'storage_before': storage_before, 'totals': dict(totals),
                        'independent_search_buffers': True}) + '\n')
            return result

    record._exact_cache_dedup = True
    processor_class._record_input_feature = record
    processor_class._layer_input_features = features
    try:
        yield totals
    finally:
        processor_class._record_input_feature = original_record
        processor_class._layer_input_features = original_features


def memory_snapshot():
    """只记录资源数字，不读取业务数据或凭据。"""
    result = {'timestamp': time.time()}
    paths = {
        'cgroup_current_bytes': ['/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory/memory.usage_in_bytes'],
        'cgroup_peak_bytes': ['/sys/fs/cgroup/memory.peak', '/sys/fs/cgroup/memory/memory.max_usage_in_bytes'],
        'cgroup_limit_bytes': ['/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'],
    }
    for key, candidates in paths.items():
        for candidate in candidates:
            try:
                value = Path(candidate).read_text().strip()
            except OSError:
                continue
            if value.isdigit() and int(value) < 2 ** 60:
                result[key] = int(value)
            break
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                key, value, _ = line.split()
                result[key.rstrip(':') + '_bytes'] = int(value) * 1024
    except OSError:
        pass
    try:
        result['cgroup_events'] = dict(line.split() for line in Path('/sys/fs/cgroup/memory.events').read_text().splitlines())
    except OSError:
        pass
    return result


@contextmanager
def memory_monitor(output, interval=10):
    """异常退出时保留内存水位；日志故障可见，不吞量化异常。"""
    stop = threading.Event()
    path = Path(output) / 'memory_usage.jsonl'

    def sample():
        try:
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(memory_snapshot()) + '\n')
        except OSError as error:
            print(f'[awq memory log] {type(error).__name__}', flush=True)

    def worker():
        while not stop.wait(interval):
            sample()

    sample()
    thread = threading.Thread(target=worker, name='awq-memory-monitor', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)
        sample()
