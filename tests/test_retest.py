"""Model reselection preserves original evidence and starts a fresh stage two."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline.common import digest, read_json, write_json
from pipeline import retest, workflow
from pipeline.launcher import resume_arguments


class RetestContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.old = self.root / 'old'
        self.new = self.root / 'new'
        self.new.mkdir()
        self.parent = self.old / 'runs/workflows/first'
        self.quant = self.old / 'runs/first-quant'
        self.accel = self.old / 'runs/inference/first-accel'
        self.model = self.quant / 'candidates/awq4/model'
        write_json(self.model / 'config.json', {})
        write_json(self.parent / 'request.json', {'quantization_run': 'first-quant', 'inference_run': 'first-accel'})
        write_json(self.parent / 'status.json', {'status': 'completed'})
        config = {'source': '/original', 'mtp_source': '/mtp', 'mtp_policy': 'preserve_original_precision',
                  'candidates': [{'id': 'original', 'recipe': 'original'}, {'id': 'awq4', 'recipe': 'compressor_awq4'}],
                  'datasets': [{'id': 'default', 'path': '/no-longer-exists'}],
                  'calibration': {'raw_data': '/no-longer-exists'},
                  'quality_gate': {'max_drop_pp': 0, 'max_regressions': 0}}
        write_json(self.quant / 'config.json', config)
        write_json(self.quant / 'status.json', {'status': 'completed'})
        write_json(self.quant / 'datasets/default.json', [{'uid': 'a', 'group': 'x', 'expected': 'A', 'scorer': 'exact'}])
        for candidate in ('original', 'awq4'):
            directory = self.quant / 'candidates' / candidate
            write_json(directory / 'status.json', {'status': 'completed'})
            path = directory / 'datasets/default/requests.jsonl'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'uid': 'a', 'group': 'x', 'content': 'A', 'status': 'success',
                                       'concurrency': 1, 'cache': 'cold', 'e2e_s': 1}) + '\n')
        write_json(self.quant / 'candidates/awq4/model_manifest.json', {
            'artifact': str(self.model), 'load_view': str(self.model), 'identity': {},
            'load_config_sha256': digest(self.model / 'config.json'), 'mtp_unchanged': True})
        write_json(self.quant / 'candidates/awq4/loading_audit.json', {'passed': True})
        write_json(self.quant / 'selected_model.json', {'candidate': 'original'})
        write_json(self.parent / 'deployment/deployment.json', {'keep': True})
        self.inference = {'runtime': {'python': '/opt/venv/bin/python', 'gpu': 0},
                          'server': {'port': 18550},
                          'benchmark': {'dataset': '/gone', 'sample_limit': 10},
                          'quality': {'dataset': '/gone', 'sample_limit': 10},
                          'search': {'max_trials': 17, 'mtp_depths': [0, 1], 'repeats': 3}}
        write_json(self.parent / 'inference.json', self.inference)
        write_json(self.accel / 'inputs.json', [{'uid': 'performance'}])
        write_json(self.accel / 'quality_inputs.json', [{'uid': 'quality'}])
        write_json(self.accel / 'status.json', {'status': 'completed'})

    def args(self, *extra):
        return workflow.parse_arguments(['--root', str(self.new), '--from-workflow', str(self.parent),
                                         '--candidate', 'awq4', '--run-id', 'second', *extra])

    def fingerprint(self):
        return {str(p): p.read_bytes() for p in self.old.rglob('*') if p.is_file()}

    def test_reselection_is_read_only_and_uses_measured_cohorts(self):
        before = self.fingerprint()
        args = self.args()
        retest.configure(args)
        self.assertTrue(args.skip_quantization)
        self.assertEqual(args.mtp_model, Path('/mtp'))
        quant, inference = workflow.build_configs(self.new, args)
        self.assertEqual(inference['search'], self.inference['search'])
        self.assertEqual(inference['benchmark']['dataset'], str(self.accel / 'inputs.json'))
        self.assertIsNone(inference['benchmark']['sample_limit'])
        # No calibration data is needed to benchmark an already measured export.
        workflow.resume_contract(self.new, args, quant, inference)
        with patch('pipeline.quantization.models.inspect_model', return_value=({}, {})):
            handoff = retest.selection(args)
        self.assertEqual(handoff['candidate'], 'awq4')
        self.assertEqual(before, self.fingerprint())

    def test_rejects_active_failed_or_modified_candidates(self):
        args = self.args()
        with patch('pipeline.jobs.status', return_value={'alive': True}):
            with self.assertRaisesRegex(ValueError, '仍在运行'):
                retest.configure(args)
        write_json(self.quant / 'candidates/awq4/status.json', {'status': 'failed'})
        with self.assertRaisesRegex(ValueError, '已完成评测'):
            retest.configure(args)
        write_json(self.quant / 'candidates/awq4/status.json', {'status': 'completed'})
        with patch('pipeline.quantization.models.inspect_model', return_value=({'changed': True}, {})):
            with self.assertRaisesRegex(ValueError, '实测模型已改变'):
                retest.selection(args)

    def test_worker_skips_quantization_remeasures_and_can_resume(self):
        args = self.args()
        retest.configure(args)
        before = self.fingerprint()
        def accelerate(root, config, model, mtp, run_id, resume=False):
            self.assertEqual(model, self.model)
            self.assertEqual(config['handoff']['candidate'], 'awq4')
            self.assertEqual(run_id, 'second-accel')
            self.assertFalse(resume)
            write_json(root / 'runs/inference' / run_id / 'status.json', {'status': 'completed'})
        with patch('pipeline.quantization.run') as quantize, \
             patch('pipeline.quantization.models.inspect_model', return_value=({}, {})), \
             patch('pipeline.inference.execution.run', side_effect=accelerate) as run, \
             patch('pipeline.text_report.report_paths'), patch('sys.stdin.isatty', return_value=False):
            directory = workflow.run(self.new, args)
            self.assertEqual(read_json(directory / 'status.json')['status'], 'awaiting_profile_selection')
            self.assertFalse(quantize.called)
            self.assertEqual(run.call_count, 1)
            continuation = resume_arguments(self.new, 'second')
            self.assertIn('--from-workflow', continuation)
            args.resume = True
            workflow.run(self.new, args)
            self.assertEqual(run.call_count, 1)
            with self.assertRaisesRegex(ValueError, "加速任务名称已存在"):
                args.resume = False
                workflow.run(self.new, args)
            self.assertEqual(before, self.fingerprint())
        # A branch can be used to pick another original candidate again.
        _, original_quant, _ = retest.source(directory)
        self.assertEqual(original_quant, self.quant)

    def test_dialog_creates_new_background_job_and_cancel_creates_nothing(self):
        try:
            from pipeline import ui, dashboard
        except ImportError:
            self.skipTest('optional terminal dependencies not installed')
        with patch.object(ui, 'show_report', return_value=[('awq4', 'AWQ4', None)]), \
             patch.object(ui, 'choose', side_effect=['awq4', 'back']), \
             patch.object(ui, 'text_input', return_value='second'), \
             patch('pipeline.jobs.submit') as submit:
            self.assertIsNone(dashboard.choose_retest(self.new, source_directory=self.parent))
            submit.assert_not_called()
        with patch.object(ui, 'show_report', return_value=[('awq4', 'AWQ4', None)]), \
             patch.object(ui, 'choose', side_effect=['awq4', 'start']), \
             patch.object(ui, 'text_input', return_value='second'), \
             patch('pipeline.jobs.submit', return_value='second') as submit:
            self.assertEqual(dashboard.choose_retest(self.new, source_directory=self.parent), 'second')
            arguments = submit.call_args.args[1]
            self.assertEqual(arguments, ['--from-workflow', str(self.parent), '--candidate', 'awq4', '--run-id', 'second'])

    def test_changed_cohort_blocks_resume(self):
        args = self.args()
        retest.configure(args)
        quant, inference = workflow.build_configs(self.new, args)
        directory = self.new / 'runs/workflows/second'
        write_json(directory / 'request.json', {'resume_contract': workflow.resume_contract(self.new, args, quant, inference)})
        write_json(directory / 'quantization.json', quant)
        write_json(directory / 'inference.json', inference)
        workflow.verify_resume(self.new, directory, args)
        write_json(self.accel / 'inputs.json', [{'uid': 'changed'}])
        with self.assertRaisesRegex(ValueError, '已变化'):
            workflow.verify_resume(self.new, directory, args)


if __name__ == '__main__':
    unittest.main()
