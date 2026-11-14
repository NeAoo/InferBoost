"""Read-only dashboard contracts: service startup, bounded logs and rotation."""
import os
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from pipeline import dashboard
from pipeline.common import write_json


class DashboardContracts(unittest.TestCase):
    def test_external_viewer_does_not_change_frozen_workspace(self):
        try:
            import rich
            import prompt_toolkit
        except ImportError:
            self.skipTest('optional terminal dependencies not installed')
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            workspace, tools = directory / 'frozen', directory / 'tools'
            shutil.copytree(project / 'pipeline', workspace / 'pipeline', ignore=shutil.ignore_patterns('__pycache__'))
            (workspace / 'pipeline/dashboard.py').write_text('raise RuntimeError("old viewer must not load")\n')
            write_json(workspace / 'jobs/demo/launch.json', {'attempt': 'test', 'log': '/missing.log'})
            write_json(workspace / 'jobs/demo/state.json', {'attempt': 'test', 'state': 'completed'})
            tools.mkdir()
            shutil.copy2(project / 'scripts/watch_live.py', tools / 'watch_live.py')
            shutil.copy2(project / 'pipeline/dashboard.py', tools / 'dashboard.py')
            def fingerprint():
                return {str(path.relative_to(workspace)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in workspace.rglob('*') if path.is_file()}
            before = fingerprint()
            result = subprocess.run([sys.executable, str(tools / 'watch_live.py'), '--root', str(workspace),
                                     '--run-id', 'demo'], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('只读', result.stdout)
            self.assertEqual(before, fingerprint())

    def test_prepared_service_has_startup_status_and_ignores_stale_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / 'runs/demo-quant/candidates/original'
            write_json(candidate / 'status.json', {'status': 'prepared'})
            job = {'state': 'running', 'workflow': {}}
            with patch('pipeline.dashboard.jobs.status', return_value=job):
                self.assertEqual(dashboard.snapshot(root, 'demo')['current'], '')
                started = time.time()
                write_json(candidate / 'process.json', {'started_at': started})
                write_json(candidate / 'cleanup.json', {'finished_at': started - 10})
                write_json(candidate / 'ready.json', {})
                os.utime(candidate / 'ready.json', (started - 5, started - 5))
                data = dashboard.snapshot(root, 'demo')
                self.assertIn('启动 vLLM', data['current'])
                self.assertEqual(data['detail_log'], str(candidate / 'server.log'))
                os.utime(candidate / 'ready.json', (started + 1, started + 1))
                self.assertIn('服务就绪', dashboard.snapshot(root, 'demo')['current'])
                write_json(candidate / 'cleanup.json', {'finished_at': started + 2})
                self.assertEqual(dashboard.snapshot(root, 'demo')['current'], '')

    def test_log_reopens_after_truncation_replacement_and_missing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'worker.log'
            self.assertIsNone(dashboard.read_log(str(path))['updated_at'])
            path.write_text('x' * (dashboard.LOG_TAIL_BYTES * 2) + '\nFIRST\n')
            initial = dashboard.read_log(str(path))
            self.assertLessEqual(len(initial['text']), dashboard.LOG_TAIL_BYTES)
            self.assertIn('FIRST', initial['text'])
            path.write_text('TRUNCATED\n')
            self.assertEqual(dashboard.read_log(str(path))['text'], 'TRUNCATED\n')
            path.rename(path.with_suffix('.old'))
            path.write_text('REPLACED\n')
            self.assertEqual(dashboard.read_log(str(path))['text'], 'REPLACED\n')
            path.unlink()
            self.assertIsNone(dashboard.read_log(str(path))['updated_at'])

    def test_terminal_controls_are_removed_but_plain_markup_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'worker.log'
            path.write_bytes(b'old\rnew\r\n\x1b[31m[red]literal[/red]\x1b[0m\n'
                             b'\x1b]0;hidden-title\x07safe\x08\n')
            log = dashboard.read_log(str(path))
            self.assertEqual(log['text'], 'new\n[red]literal[/red]\nsafe\n')
            try:
                import rich
            except ImportError:
                self.skipTest('optional terminal dependencies not installed')
            data = {'job': {'state': 'running', 'log': str(path)}, 'current': '量化'}
            output = dashboard.render_log(data, log, 'demo', width=60, height=14, colors=False)
            self.assertIn('[red]literal[/red]', output)
            self.assertNotIn('\x1b', output)
            self.assertIn('实时日志', output)
            self.assertLessEqual(len(output.splitlines()), 14)

    def test_log_keeps_only_latest_fifty_rows_and_fits_short_terminal(self):
        try:
            import rich
        except ImportError:
            self.skipTest('optional terminal dependencies not installed')
        data = {'job': {'state': 'running', 'log': '/worker.log', 'workflow': {}},
                'current': 'AWQ4', 'progress': {},
                'counts': dict(completed=1, failed=0, skipped=0, prepared=0)}
        log = {'path': '/worker.log', 'text': '\n'.join(f'LINE_{i:03d}' for i in range(101)),
               'updated_at': time.time()}
        output = dashboard.render_log(data, log, 'demo', width=80, height=70, colors=False)
        self.assertNotIn('LINE_050', output)
        self.assertIn('LINE_051', output)
        self.assertIn('LINE_100', output)
        self.assertEqual(sum(line.startswith('LINE_') for line in output.splitlines()), 50)
        for height in (3, 8, 15, 24):
            output = dashboard.render_log(data, log, 'demo', width=40, height=height, colors=False)
            self.assertLessEqual(len(output.splitlines()), height)
            self.assertIn('LINE_100', output)
        log['text'] = 'long=' + 'x' * 10000 + '\nLATEST'
        output = dashboard.render_log(data, log, 'demo', width=40, height=8, colors=False)
        self.assertIn('LATEST', output)
        self.assertEqual(len(output.splitlines()), 4)
        for height in (7, 9):
            header = dashboard.render_pinned(data, {}, 'demo', width=40, height=height, colors=False)
            self.assertTrue(header.startswith('InferBoost'))
            self.assertIn('AWQ4', header)
            self.assertLessEqual(len(header.splitlines()), height)


if __name__ == '__main__':
    unittest.main()
