"""Cooperative per-GPU leases shared by quantization, tuning and deployment."""
from contextlib import contextmanager
import fcntl
from pathlib import Path
import re
import subprocess
import time


@contextmanager
def gpu_lease(root, gpu=0, wait_seconds=0):
    root = Path(root)
    identity = subprocess.check_output(
        ['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid', '--format=csv,noheader'],
        text=True, timeout=20).strip()
    if not re.fullmatch(r'GPU-[A-Za-z0-9-]+', identity):
        raise ValueError('无法确定唯一GPU身份: ' + identity)
    locks = root / 'runtime/locks'
    locks.mkdir(parents=True, exist_ok=True)
    # Legacy coordinators take an exclusive global lease; new ones share it and
    # additionally lease a physical GPU, so old and new processes cannot collide.
    with (root / 'gpu.lock').open('a') as legacy, (locks / (identity + '.lock')).open('a') as device:
        started = time.monotonic()
        while True:
            try:
                fcntl.flock(legacy, fcntl.LOCK_SH | fcntl.LOCK_NB)
                fcntl.flock(device, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                fcntl.flock(legacy, fcntl.LOCK_UN)
                if time.monotonic() - started >= wait_seconds:
                    raise RuntimeError(f'GPU {gpu} 正被其他任务使用')
                time.sleep(min(2, max(.1, wait_seconds - (time.monotonic() - started))))
        yield identity
