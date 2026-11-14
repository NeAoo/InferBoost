"""Contracts for real worker lifecycle and measured UI data; no GPU required."""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from pipeline import jobs, measurement, preflight
from pipeline.common import write_json
from pipeline.modes import apply_mode, sample_rows
from pipeline.workflow import parse_arguments, build_configs

ROOT = Path(__file__).resolve().parents[1]


class PresetContracts(unittest.TestCase):
    def test_standard_and_quick_have_explicit_distinct_scopes(self):
        for mode, repeats, count, limit in [('standard', 3, 512, None), ('quick', 1, 64, 10)]:
            args = parse_arguments(['--root', str(ROOT), '--model', '/model', '--mode', mode, '--recipes', 'gptq4'])
            quant, inference = build_configs(ROOT, args)
            self.assertEqual(quant['calibration']['count'], count)
            self.assertEqual(quant['benchmark']['performance_repeats'], repeats)
            self.assertEqual(quant['quality_gate'], {'max_drop_pp': 5.0, 'max_regressions': 3})
            self.assertTrue(quant['benchmark']['separate_quality'])
            self.assertEqual(inference['benchmark']['sample_limit'], limit)
            self.assertEqual(quant['experience']['smoke_only'], mode == 'quick')
            self.assertEqual(inference['search']['repeats'], repeats)

    def test_invalid_budget_does_not_create_task(self):
        args = parse_arguments(['--root', str(ROOT), '--model', '/model', '--calibration-count', '513'])
        with self.assertRaisesRegex(ValueError, 'calibration_count'):
            build_configs(ROOT, args)

    def test_quick_sampling_is_deterministic_and_group_balanced(self):
        rows = [{'uid': str(n), 'group': group} for group in ('a', 'b') for n in range(10)]
        selected = sample_rows(rows, 4)
        self.assertEqual([r['group'] for r in selected], ['a', 'b', 'a', 'b'])
        self.assertEqual(selected, sample_rows(rows, 4))

    def test_mount_detection_does_not_claim_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            info = Path(directory) / 'mountinfo'
            info.write_text('20 1 0:1 / / rw - overlay overlay rw\n30 1 0:3 / /vepfs rw - nfs server:/data rw\n')
            root = preflight.mount_info('/tmp/run', info)
            volume = preflight.mount_info('/vepfs/user/run', info)
            self.assertIn('临时', root['persistence'])
            self.assertIn('仍需平台确认', volume['persistence'])
            self.assertEqual(volume['mount'], '/vepfs')

    def test_admission_matches_existing_moe_restrictions(self):
        model = {'adapter': {'family': 'qwen35_moe', 'moe': True},
                 'config': {'model_type': 'qwen3_5_moe'}}
        rows = {r['id']: r for r in preflight.recipe_choices(model, [{'capability': [8, 0]}])}
        self.assertTrue(rows['awq4']['enabled'])
        self.assertFalse(rows['fp8']['enabled'])
        self.assertTrue(rows['gptq4']['enabled'])


class MeasurementContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rows = [{'uid': 'a', 'group': 'x'}, {'uid': 'b', 'group': 'x'}]
        self.settings = {'performance_repeats': 3, 'concurrencies': [1], 'cache_modes': ['cold', 'warm']}

    def records(self, index, missing=False):
        path = self.root / 'performance' / f'round-{index}' / 'requests.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [dict(row, cache=cache, concurrency=1, status='success',
                        e2e_s=index, ttft_s=index / 10, decode_tps=100 / index)
                   for cache in ['cold', 'warm'] for row in self.rows]
        path.write_text(''.join(json.dumps(r) + '\n' for r in (records[:-1] if missing else records)))

    def test_three_round_means_and_standard_deviation(self):
        for index in range(1, 4):
            self.records(index)
        result = measurement.performance_summary(self.root, self.rows, self.settings)
        self.assertTrue(result['complete'])
        self.assertEqual(result['e2e_s'], 2)
        self.assertEqual(result['e2e_s_std'], 1)
        self.assertEqual(result['unique_samples'], 2)
        self.assertEqual(result['requests_per_cache_c1'], 6)

    def test_missing_warm_round_excludes_mean(self):
        for index in range(1, 4):
            self.records(index, missing=index == 3)
        result = measurement.performance_summary(self.root, self.rows, self.settings)
        self.assertFalse(result['complete'])
        self.assertIsNone(result['e2e_s'])

    def test_quality_once_and_performance_three_times_in_different_files(self):
        with patch('pipeline.benchmark.run_suite') as suite, patch('pipeline.runtime.warmup') as warmup:
            measurement.measure('unused', self.rows, self.settings, self.root, 'gptq4')
        self.assertEqual(suite.call_count, 4)
        self.assertEqual(warmup.call_count, 3)
        self.assertEqual(suite.call_args_list[0].args[2]['cache_modes'], ['cold'])
        self.assertEqual(suite.call_args_list[0].args[3], self.root / 'quality')
        self.assertEqual(suite.call_args_list[-1].args[3], self.root / 'performance/round-3')


FAKE_WORKFLOW = '''
import time
from pathlib import Path
from pipeline.common import write_json
def main(arguments, install_signals=True):
    root=Path(arguments[arguments.index('--root')+1])
    name=arguments[arguments.index('--run-id')+1]
    path=root/'runs/workflows'/name
    write_json(path/'status.json', {'status':'quantization'})
    write_json(path/'request.json', {'invocation':{'run_id':name,'model':'/test-model'}})
    try:
        for tick in range(120):
            write_json(path/'tick.json', {'tick':tick})
            time.sleep(.1)
        write_json(path/'status.json', {'status':'awaiting_model_selection'})
    finally:
        write_json(path/'cleanup.json', {'called':True})
'''


class WorkerContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        shutil.copytree(ROOT / 'pipeline', self.root / 'pipeline', ignore=shutil.ignore_patterns('__pycache__'))
        (self.root / 'pipeline/workflow.py').write_text(FAKE_WORKFLOW)
        self.run_id = 'lifecycle'

    def tearDown(self):
        current = jobs.status(self.root, self.run_id)
        if current['alive']:
            jobs.request_stop(self.root, self.run_id)
            self.wait_for(lambda: not jobs.status(self.root, self.run_id)['alive'])
        # Reap a direct child when submit was called in this test process.
        if current.get('pid'):
            try:
                os.waitpid(current['pid'], os.WNOHANG)
            except ChildProcessError:
                pass
        self.temporary.cleanup()

    def wait_for(self, condition, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(.05)
        self.fail('worker state did not arrive within timeout')

    def submit(self):
        jobs.submit(self.root, ['--run-id', self.run_id, '--model', '/test-model'])
        self.wait_for(lambda: jobs.status(self.root, self.run_id).get('state') == 'running')

    def test_worker_survives_submitter_exit_and_explicit_stop_cleans_up(self):
        code = ('from pathlib import Path; from pipeline.jobs import submit; '
                'submit(Path(' + repr(str(self.root)) + '), ["--run-id","lifecycle","--model","/test-model"])')
        subprocess.run([sys.executable, '-c', code], cwd=self.root,
                       env=dict(os.environ, PYTHONPATH=str(self.root)), check=True, timeout=10)
        tick = self.root / 'runs/workflows/lifecycle/tick.json'
        self.wait_for(tick.exists)
        first = jobs.read_optional(tick).get('tick', -1)
        self.wait_for(lambda: jobs.read_optional(tick).get('tick', -1) > first + 1)
        self.assertTrue(jobs.status(self.root, self.run_id)['alive'])
        jobs.request_stop(self.root, self.run_id)
        self.wait_for(lambda: jobs.status(self.root, self.run_id).get('state') == 'interrupted')
        self.assertTrue((tick.parent / 'cleanup.json').is_file())

    def test_duplicate_submit_rejected_and_stale_stop_ignored(self):
        self.submit()
        with self.assertRaisesRegex(RuntimeError, '仍在运行'):
            jobs.submit(self.root, ['--run-id', self.run_id])
        write_json(self.root / 'jobs/lifecycle/stop.json', {'attempt': 'old-attempt'})
        time.sleep(1.1)
        self.assertTrue(jobs.status(self.root, self.run_id)['alive'])

    def test_wrong_process_identity_is_not_live(self):
        self.assertFalse(jobs.alive({'pid': os.getpid(), 'identity': 'not-this-process'}))

    def test_quit_dashboard_keeps_real_worker_running(self):
        try:
            from prompt_toolkit.input import create_pipe_input
            from prompt_toolkit.output import DummyOutput
            from prompt_toolkit.application import create_app_session
            from pipeline.dashboard import watch
            import rich
        except ImportError:
            self.skipTest('optional terminal dependencies not installed')
        self.submit()
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), \
                patch('sys.stdin.isatty', return_value=True):
            pipe.send_text('q')
            watch(self.root, self.run_id)
        self.assertTrue(jobs.status(self.root, self.run_id)['alive'])
        self.assertFalse((self.root / 'jobs/lifecycle/stop.json').exists())

    def test_live_logs_append_switch_stage_and_quit_without_stopping_worker(self):
        try:
            from prompt_toolkit import Application
            from prompt_toolkit.input import create_pipe_input
            from prompt_toolkit.output import DummyOutput
            from prompt_toolkit.application import create_app_session
            from pipeline import dashboard
            import rich
        except ImportError:
            self.skipTest('optional terminal dependencies not installed')
        self.submit()
        main_log = Path(jobs.status(self.root, self.run_id)['log'])
        main_log.write_text('FIRST_LOG_LINE\n')
        candidate = self.root / 'runs/lifecycle-quant/candidates/awq4'
        write_json(candidate / 'status.json', {'status': 'quantizing_or_loading'})
        (candidate / 'quantization.log').write_text('QUANT_LAYER_PROGRESS\n')
        markers = ['FIRST_LOG_LINE', 'APPENDED_WHILE_VIEWING', 'QUANT_LAYER_PROGRESS', 'SERVER_LOADING_NOW']
        observed = {marker: threading.Event() for marker in markers}
        original_render = dashboard.render_log
        rendered_screens = []
        release_resources = threading.Event()
        def slow_resources(*args):
            release_resources.wait(5)
            return {}
        def application(*args, **kwargs):
            app = Application(*args, **kwargs)
            self.assertTrue(app.full_screen)
            def screen_after_render(sender):
                screen = sender.renderer._last_screen
                if screen is not None:
                    rows = [''.join(row[column].char for column in range(80))
                            for row in [screen.data_buffer[index] for index in range(40)]]
                    if any('APPENDED_WHILE_VIEWING' in row for row in rows):
                        rendered_screens.append(rows)
            app.after_render += screen_after_render
            return app
        def capture(*args, **kwargs):
            output = original_render(*args, **kwargs)
            for marker, event in observed.items():
                if marker in output:
                    event.set()
            return output
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), \
                patch('sys.stdin.isatty', return_value=True), \
                patch('pipeline.dashboard.REFRESH_SECONDS', .03), \
                patch('pipeline.dashboard.preflight.inspect_resources', side_effect=slow_resources), \
                patch('prompt_toolkit.Application', side_effect=application), \
                patch('pipeline.dashboard.render_log', side_effect=capture), \
                patch('pipeline.ui.text_input', side_effect=AssertionError('日志视图不能等待静态输入')):
            def interact():
                try:
                    pipe.send_text('l')
                    if not observed['FIRST_LOG_LINE'].wait(3):
                        return
                    with main_log.open('a') as stream:
                        stream.write('APPENDED_WHILE_VIEWING\n')
                    if not observed['APPENDED_WHILE_VIEWING'].wait(3):
                        return
                    pipe.send_text('\t')
                    if not observed['QUANT_LAYER_PROGRESS'].wait(3):
                        return
                    (candidate / 'server.log').write_text('SERVER_LOADING_NOW\n')
                    write_json(candidate / 'process.json', {'started_at': time.time()})
                    write_json(candidate / 'status.json', {'status': 'prepared'})
                    observed['SERVER_LOADING_NOW'].wait(3)
                finally:
                    release_resources.set()
                    pipe.send_text('q')
            thread = threading.Thread(target=interact, daemon=True)
            thread.start()
            dashboard.watch(self.root, self.run_id)
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(all(event.is_set() for event in observed.values()), str(observed))
        self.assertTrue(rendered_screens)
        self.assertTrue(all('InferBoost' in rows[0] for rows in rendered_screens))
        self.assertTrue(jobs.status(self.root, self.run_id)['alive'])
        self.assertFalse((self.root / 'jobs/lifecycle/stop.json').exists())

    def test_read_only_viewer_ignores_stop_and_resume_keys(self):
        try:
            from prompt_toolkit.input import create_pipe_input
            from prompt_toolkit.output import DummyOutput
            from prompt_toolkit.application import create_app_session
            from pipeline.dashboard import watch
            import rich
        except ImportError:
            self.skipTest('optional terminal dependencies not installed')
        self.submit()
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), \
                patch('sys.stdin.isatty', return_value=True), \
                patch('pipeline.jobs.request_stop') as stop, patch('pipeline.jobs.submit') as submit:
            pipe.send_text('scq')
            watch(self.root, self.run_id, read_only=True)
        stop.assert_not_called()
        submit.assert_not_called()
        self.assertTrue(jobs.status(self.root, self.run_id)['alive'])

    def test_continuation_keeps_budget_and_manual_selection(self):
        invocation = {'run_id': self.run_id, 'model': '/test-model', 'mode': 'custom',
                      'performance_repeats': 3, 'calibration_count': 128}
        write_json(self.root / 'runs/workflows/lifecycle/request.json', {'invocation': invocation})
        write_json(self.root / 'jobs/lifecycle/request.json', {'arguments': ['--candidate', 'gptq4']})
        from pipeline.jobs import continuation
        args = continuation(self.root, self.run_id, profile='chosen')
        self.assertEqual(jobs.argument_value(args, '--candidate'), 'gptq4')
        self.assertEqual(jobs.argument_value(args, '--performance-repeats'), '3')
        self.assertEqual(jobs.argument_value(args, '--profile'), 'chosen')


class RichInteractionContracts(unittest.TestCase):
    def test_choice_keyboard_filter_and_disabled_options(self):
        try:
            from prompt_toolkit.input import create_pipe_input
            from prompt_toolkit.output import DummyOutput
            from prompt_toolkit.application import create_app_session
            from pipeline.ui import choose
        except ImportError:
            self.skipTest('optional terminal dependencies not installed')
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            pipe.send_text('fp8\r\x7f\x7f\x7fgptq\r')
            value = choose('recipes', [('fp8', 'FP8', 'SM80不支持'), ('gptq', 'GPTQ', None)])
        self.assertEqual(value, 'gptq')

    def test_multiselect_space_and_enter(self):
        try:
            from prompt_toolkit.input import create_pipe_input
            from prompt_toolkit.output import DummyOutput
            from prompt_toolkit.application import create_app_session
            from pipeline.ui import choose
        except ImportError:
            self.skipTest('optional terminal dependencies not installed')
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            pipe.send_text(' \x1b[B \r')
            values = choose('recipes', [('gptq', 'GPTQ', None), ('awq', 'AWQ', None)], multiple=True)
        self.assertEqual(values, ['gptq', 'awq'])

    def test_dashboard_renders_unknown_and_failed_without_fake_percent(self):
        try:
            from pipeline.dashboard import render
            import rich
        except ImportError:
            self.skipTest('optional terminal dependencies not installed')
        data = {'job': {'state': 'failed', 'error': 'exit 137', 'workflow': {}},
                'counts': dict(completed=1, failed=1, skipped=2, prepared=0), 'progress': {}, 'current': '量化'}
        display = render(data, {}, 'test', width=80, height=40, colors=False)
        self.assertIn('失败', display)
        self.assertIn('不能单独证明OOM', display)
        self.assertNotIn('100%', display)
        self.assertNotIn('\x1b', display)


if __name__ == '__main__':
    unittest.main()
