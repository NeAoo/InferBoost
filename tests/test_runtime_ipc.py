import os
from pathlib import Path
import socket
import stat
import tempfile
import unittest
from unittest.mock import patch
import uuid

from pipeline.common import runtime_environment
from pipeline.ipc import socket_directory


class RuntimeSocketPaths(unittest.TestCase):
    def test_deep_workspace_binds_a_real_unix_socket_without_relocating_cache(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            root = Path(temporary) / ('persistent-model-experiments-' * 6)
            ipc_root = Path(temporary) / 'ipc'
            ipc_root.mkdir()
            with patch('pipeline.ipc.IPC_TEMP_ROOT', ipc_root), patch.dict(os.environ, {}, clear=True):
                environment = runtime_environment(root, {'runtime': {'gpu': 0}})
                address = Path(environment['VLLM_RPC_BASE_PATH']) / str(uuid.uuid4())
                with socket.socket(socket.AF_UNIX) as connection:
                    connection.bind(str(address))
                self.assertEqual(environment['TMPDIR'], str((root / 'tmp').resolve()))
                self.assertEqual(environment['VLLM_CACHE_ROOT'], str((root / 'cache/vllm').resolve()))
                self.assertLess(len(os.fsencode(address)), 104)
                self.assertEqual(stat.S_IMODE(address.parent.stat().st_mode), 0o700)

    def test_workspaces_are_separate_and_an_explicit_override_is_preserved(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            with patch('pipeline.ipc.IPC_TEMP_ROOT', Path(temporary)), patch.dict(os.environ, {}, clear=True):
                first, second = Path(temporary) / 'a', Path(temporary) / 'b'
                self.assertNotEqual(socket_directory(first), socket_directory(second))
                custom = str(Path(temporary) / 'explicit')
                environment = runtime_environment(first, {'runtime': {'environment': {'VLLM_RPC_BASE_PATH': custom}}})
                self.assertEqual(environment['VLLM_RPC_BASE_PATH'], custom)

    def test_existing_symlink_is_not_used_as_a_socket_directory(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            with patch('pipeline.ipc.IPC_TEMP_ROOT', Path(temporary)):
                root = Path(temporary) / 'workspace'
                directory = Path(socket_directory(root))
                directory.rmdir()
                directory.symlink_to(temporary, target_is_directory=True)
                with self.assertRaises(ValueError):
                    socket_directory(root)


if __name__ == '__main__':
    unittest.main()
