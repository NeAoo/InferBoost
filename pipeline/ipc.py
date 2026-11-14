"""Keep local vLLM sockets short even when persistent workspaces are deeply nested."""
import hashlib
import os
from pathlib import Path
import stat

IPC_TEMP_ROOT = Path('/tmp')


def socket_directory(root):
    # vLLM appends a UUID to VLLM_RPC_BASE_PATH. Linux sockaddr_un permits only
    # 107 path bytes. Compilation/temp data still live in the persistent workspace.
    workspace_id = hashlib.sha256(os.fsencode(Path(root).resolve())).hexdigest()[:16]
    directory = IPC_TEMP_ROOT / f'inferboost-ipc-{os.getuid()}-{workspace_id}'
    directory.mkdir(mode=0o700, exist_ok=True)
    attributes = directory.lstat()
    if (not stat.S_ISDIR(attributes.st_mode) or attributes.st_uid != os.getuid()
            or stat.S_IMODE(attributes.st_mode) & 0o077):
        raise ValueError('vLLM通信目录必须由当前用户独占且不能是链接: ' + str(directory))
    return str(directory)
