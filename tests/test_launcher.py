import argparse
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from pipeline import launcher
from pipeline.common import write_json
from pipeline.inference.defaults import defaults_html, defaults_markdown
from pipeline.inference.decision_trace import build
from pipeline.inference.decision_view import render
from pipeline.workflow import main as workflow_main


class LauncherContracts(unittest.TestCase):
    def test_menu_does_not_start_work_when_exiting_or_without_tty(self):
        with tempfile.TemporaryDirectory() as temporary, patch('sys.stdout', new_callable=io.StringIO) as output:
            with patch('sys.stdin.isatty', return_value=False), patch.object(launcher, 'run_workflow') as run:
                launcher.main(['--root', temporary])
                run.assert_not_called()
            self.assertIn('InferBoost', output.getvalue())
            self.assertIn('需要交互终端', output.getvalue())
            with patch('sys.stdin.isatty', return_value=True), patch('builtins.input', return_value='0'), \
                    patch.object(launcher, 'run_workflow') as run:
                launcher.main(['--root', temporary, '--plain'])
                run.assert_not_called()

    def test_menu_preserves_spaces_and_requires_user_selection_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / 'model with spaces'
            write_json(model / 'config.json', {'model_type': 'qwen3_5'})
            write_json(root / 'configs/inference.example.json', {'server': {'max_model_len': 32768}})
            answers = [str(model), 'same', 'awq4', '', '0, 1', '', '', 'first-run', '']
            with patch('builtins.input', side_effect=answers), patch('sys.stdout', new_callable=io.StringIO) as output:
                arguments = launcher.workflow_arguments(root, False)
            self.assertEqual(arguments[arguments.index('--model') + 1], str(model.resolve()))
            self.assertEqual(arguments[arguments.index('--mtp-model') + 1], str(model.resolve()))
            self.assertEqual(arguments[arguments.index('--gpus') + 1], '0,1')
            self.assertNotIn('--auto-select', arguments)
            self.assertIn('quantized', output.getvalue())
            self.assertIn('普通KV Cache', output.getvalue())

    def test_quantized_model_is_not_requantized_and_invalid_paths_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / 'config.json', {'quantization_config': {'quant_method': 'compressed-tensors'}})
            with self.assertRaisesRegex(ValueError, '已有量化'):
                launcher.model_input(str(root), False)
            self.assertEqual(launcher.model_input('"' + str(root) + '"', True), root.resolve())
            with self.assertRaisesRegex(ValueError, 'config.json'):
                launcher.model_input(str(root / 'missing'), True)
            with self.assertRaises(ValueError):
                launcher.normalize_gpus('0,0')

    def test_resume_uses_saved_arguments_and_refuses_different_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            invocation = {'model': '/models/has space', 'run_id': 'example', 'gpus': '0,1',
                          'recipes': 'awq4', 'auto_select': False, 'skip_quantization': True, 'max_jobs': 2}
            path = root / 'runs/workflows/example/request.json'
            write_json(path, {'invocation': invocation})
            arguments = launcher.resume_arguments(root, 'example')
            self.assertIn('--resume', arguments)
            self.assertIn('--skip-quantization', arguments)
            self.assertNotIn('--auto-select', arguments)
            self.assertEqual(arguments[arguments.index('--model') + 1], '/models/has space')
            invocation['run_id'] = 'other'
            write_json(path, {'invocation': invocation})
            with self.assertRaisesRegex(ValueError, '不一致'):
                launcher.resume_arguments(root, 'example')

    def test_confirmed_menu_uses_same_workflow_and_cancel_does_not_start(self):
        with tempfile.TemporaryDirectory() as temporary, patch('sys.stdout', new_callable=io.StringIO):
            root = Path(temporary)
            with patch('builtins.input', return_value='n'), patch('pipeline.workflow.main') as run:
                launcher.run_workflow(root, ['--model', '/models/test'])
                run.assert_not_called()
            with patch('builtins.input', return_value='y'), patch('pipeline.workflow.main') as run:
                launcher.run_workflow(root, ['--model', '/models/test'])
                run.assert_called_once_with(['--root', str(root), '--model', '/models/test'], install_signals=False)

    def test_defaults_appear_before_choices_and_only_once_in_html(self):
        evidence = {'decision_steps': [], 'records': [], 'objective_cache': 'warm', 'skipped': [],
                    'screened_candidates': 0, 'complete_screened_candidates': 0, 'recorded_common_settings_by_profile': {}}
        document = render(evidence, 'GPTQ-W8A8')
        self.assertEqual(document.count('id="defaults"'), 1)
        self.assertLess(document.index('普通KV Cache'), document.index('先读路线'))
        self.assertIn('没有分别做开关消融', defaults_markdown())
        self.assertIn('auto', defaults_html())

    def test_workflow_cli_and_noninteractive_help_remain_compatible(self):
        with tempfile.TemporaryDirectory() as temporary, patch('pipeline.workflow.run') as run:
            workflow_main(['--root', temporary, '--model', '/models/test', '--recipes', 'awq4',
                           '--run-id', 'test', '--max-jobs', '2'], install_signals=False)
            parsed = run.call_args.args[1]
            self.assertFalse(parsed.auto_select)
            self.assertEqual(parsed.model, Path('/models/test'))
            self.assertEqual(parsed.max_jobs, 2)
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([str(root / 'quickstart'), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--model', result.stdout)

    def test_history_reports_actual_reused_weight_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'runs/example-quant'
            write_json(run / 'status.json', {'status': 'completed'})
            artifact = root / 'older-export'
            artifact.mkdir()
            write_json(run / 'candidates/awq4/model_manifest.json', {'artifact': str(artifact), 'load_view': str(run / 'candidates/awq4/model')})
            with patch('builtins.input', return_value='1'), patch('sys.stdout', new_callable=io.StringIO) as output:
                launcher.history(root)
            self.assertIn(str(artifact), output.getvalue())
            self.assertIn('不要只复制此目录', output.getvalue())


if __name__ == '__main__':
    unittest.main()
