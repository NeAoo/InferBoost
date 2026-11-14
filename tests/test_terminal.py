import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from pipeline import launcher, terminal, benchmark
from pipeline.common import write_json
from pipeline.inference.execution import wait_with_progress


class Tty(io.StringIO):
    def isatty(self):
        return True


class WizardNavigation(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / 'model'
        write_json(self.model / 'config.json', {'model_type': 'qwen3'})
        write_json(self.root / 'configs/inference.example.json', {'server': {'max_model_len': 32768}})

    def collect(self, answers, skip=False):
        with patch('builtins.input', side_effect=answers), patch('sys.stdout', new_callable=io.StringIO):
            return launcher.WorkflowWizard(self.root, skip).collect()

    def test_back_retains_other_answers_and_updates_resource_argument(self):
        args = self.collect([str(self.model), '', 'awq4', '', '0', ':b', '0,1', '4', '', 'back-test', ''])
        self.assertEqual(args[args.index('--gpus') + 1], '0,1')
        self.assertEqual(args[args.index('--max-jobs') + 1], '4')
        self.assertNotIn('--auto-select', args)
        self.assertFalse((self.root / 'runs').exists())

    def test_same_mtp_tracks_model_changed_after_going_back(self):
        other = self.root / 'other model'
        write_json(other / 'config.json', {'model_type': 'qwen3'})
        args = self.collect([str(self.model), 'same', ':b', ':b', str(other), '', 'awq4', '', '0', '', '', 'new-model', ''])
        self.assertEqual(args[args.index('--model') + 1], str(other.resolve()))
        self.assertEqual(args[args.index('--mtp-model') + 1], str(other.resolve()))

    def test_optional_mtp_can_be_cleared_without_losing_model(self):
        args = self.collect([str(self.model), 'same', ':b', ':clear', 'awq4', '', '0', '', '', 'no-mtp', ''])
        self.assertNotIn('--mtp-model', args)
        self.assertEqual(args[1], str(self.model.resolve()))

    def test_invalid_input_stays_at_same_question_and_cancel_creates_nothing(self):
        with patch('builtins.input', side_effect=[':b', '/missing/model', ':q']), \
                patch('sys.stdout', new_callable=io.StringIO) as output:
            with self.assertRaises(launcher.WizardCancelled):
                launcher.WorkflowWizard(self.root, False).collect()
        self.assertIn('已经是第一步', output.getvalue())
        self.assertIn('之前的输入已保留', output.getvalue())
        self.assertFalse((self.root / 'runs').exists())

    def test_prequantized_flow_omits_recipe_question(self):
        args = self.collect([str(self.model), '', '', '0', '', '', 'accel-only', 'y'], skip=True)
        self.assertIn('--skip-quantization', args)
        self.assertIn('--auto-select', args)
        self.assertNotIn('--recipes', args)

    def test_confirmation_back_does_not_launch_gpu_work(self):
        with patch('builtins.input', return_value=':b'), patch('sys.stdout', new_callable=io.StringIO), \
                patch('pipeline.workflow.main') as run:
            self.assertEqual(launcher.run_workflow(self.root, ['--model', str(self.model)], allow_back=True), 'back')
            run.assert_not_called()


class ProgressContracts(unittest.TestCase):
    def test_color_respects_no_color_and_noninteractive_output(self):
        with patch.dict(os.environ, {'TERM': 'xterm-256color'}, clear=True):
            self.assertIn('\x1b[', terminal.style('InferBoost', stream=Tty()))
            self.assertNotIn('\x1b', terminal.style('InferBoost', stream=io.StringIO()))
            with patch.dict(os.environ, {'NO_COLOR': ''}):
                self.assertNotIn('\x1b', terminal.style('InferBoost', stream=Tty()))

    def test_interrupted_bar_keeps_real_count_and_never_fills_to_100(self):
        output = io.StringIO()
        with self.assertRaises(KeyboardInterrupt):
            with terminal.Progress('评测', 100, stream=output) as progress:
                progress.update(37, force=True)
                raise KeyboardInterrupt()
        text = output.getvalue()
        self.assertIn('37/100', text)
        self.assertIn('已中断', text)
        self.assertNotIn('(100%)', text)
        self.assertNotIn('\x1b', text)

    def test_unknown_total_shows_elapsed_without_percentage(self):
        output = io.StringIO()
        with terminal.Progress('编译', stream=output):
            pass
        self.assertIn('已用', output.getvalue())
        self.assertNotIn('%', output.getvalue())

    def test_narrow_terminal_preserves_cjk_character_boundaries(self):
        self.assertEqual(terminal.width_limit('中文 abc', 5), '中文…')
        self.assertNotIn('\x1b', terminal.clean('\x1b[31mhello\r\n'))

    def test_measured_request_counts_include_each_cache_and_keep_artifacts(self):
        rows = [{'uid': 'x', 'group': 'g', 'messages': []}]
        settings = {'concurrencies': [1, 2], 'cache_modes': ['cold', 'warm']}
        with tempfile.TemporaryDirectory() as temporary, patch('sys.stdout', new_callable=io.StringIO) as output, \
                patch.object(benchmark, 'endpoint', return_value='metrics'), patch.object(benchmark, 'reset_cache'), \
                patch.object(benchmark, 'request_one', side_effect=lambda *_: {'status': 'success'}) as request:
            benchmark.run_suite('unused', rows, settings, Path(temporary), 'awq4')
            self.assertEqual(request.call_count, 4)
            self.assertIn('4/4 请求 (100%)', output.getvalue())
            self.assertEqual(len((Path(temporary) / 'requests.jsonl').read_text().splitlines()), 4)

    def test_failed_sweep_is_not_displayed_as_complete(self):
        with tempfile.TemporaryDirectory() as temporary, patch('sys.stdout', new_callable=io.StringIO) as output:
            run = Path(temporary)
            write_json(run / 'active.json', {'phase': 'attention', 'profile_id': 'fa2'})
            write_json(run / 'progress.json', {'phase': 'attention', 'profile_id': 'fa2', 'profile': 'FA2',
                       'round': 'run=0.json', 'stage': 'performance', 'completed': 40, 'total': 100})
            class Process:
                calls = 0
                def wait(self, timeout):
                    self.calls += 1
                    if self.calls == 1:
                        raise subprocess.TimeoutExpired('worker', timeout)
                    return 1
            self.assertEqual(wait_with_progress(Process(), run, 'attention'), 1)
            self.assertIn('40/100', output.getvalue())
            self.assertIn('失败', output.getvalue())
            self.assertNotIn('(100%)', output.getvalue())


if __name__ == '__main__':
    unittest.main()
