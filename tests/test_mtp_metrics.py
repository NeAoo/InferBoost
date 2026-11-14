import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from pipeline.common import read_json, write_json
from pipeline.inference import mtp_metrics as metrics, worker
from pipeline.inference.execution import publish_phase, wait_with_progress
from pipeline.inference.planning import make_trial


def snapshot(drafts, drafted, accepted, engine='0'):
    raw = '\n'.join(f'{name}{{model_name="model",engine="{engine}"}} {value}'
                    for name, value in zip(metrics.COUNTERS.values(), (drafts, drafted, accepted)))
    return {'status': 'captured', 'series': metrics.parse(raw)}


class AcceptanceMetrics(unittest.TestCase):
    def test_weighted_counts_not_average_percentages_and_bonus_definition(self):
        first = metrics.difference(snapshot(10, 20, 8), snapshot(20, 40, 28))
        second = metrics.difference(snapshot(0, 0, 0), snapshot(90, 180, 90))
        merged = metrics.summarize([{'mtp_acceptance': first}, {'mtp_acceptance': second}], 2, '/absent', 2)
        self.assertEqual(merged['accepted_tokens'], 110)
        self.assertEqual(merged['drafted_tokens'], 200)
        self.assertAlmostEqual(merged['acceptance_rate'], .55)
        self.assertAlmostEqual(merged['mean_acceptance_length_with_bonus'], 2.1)

    def test_missing_reset_labels_zero_and_invalid_counts(self):
        self.assertEqual(metrics.difference(snapshot(10, 20, 10), snapshot(1, 2, 1))['status'], 'invalid')
        self.assertEqual(metrics.difference(snapshot(1, 2, 1), snapshot(1, 2, 1, '1'))['status'], 'invalid')
        self.assertEqual(metrics.difference(snapshot(1, 2, 1), metrics.unavailable('missing'))['status'], 'unavailable')
        self.assertIsNone(metrics.difference(snapshot(0, 0, 0), snapshot(0, 0, 0))['acceptance_rate'])
        self.assertEqual(metrics.counts(3, 2)['status'], 'invalid')
        self.assertEqual(metrics.counts(1, 2, 0)['status'], 'invalid')
        self.assertEqual(metrics.counts(0, 2, 1)['acceptance_rate'], 0)
        with self.assertRaises(ValueError):
            metrics.parse('vllm:spec_decode_num_drafts_total 1')
        with self.assertRaises(ValueError):
            metrics.parse('vllm:spec_decode_num_drafts_total NaN')

    def test_multiple_engines_cannot_hide_one_counter_reset(self):
        first, last = snapshot(10, 20, 10), snapshot(9, 18, 9)
        for key in metrics.COUNTERS:
            first['series'][key].update(snapshot(10, 20, 10, '1')['series'][key])
            last['series'][key].update(snapshot(30, 60, 30, '1')['series'][key])
        self.assertEqual(metrics.difference(first, last)['status'], 'invalid')

    def test_log_fallback_is_explicit_and_disabled_is_not_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / 'server.log'
            log.write_text('warning draft\nSpecDecoding metrics: Accepted: 1 tokens, Drafted: 2 tokens\n'
                           'SpecDecoding metrics: Accepted: 9 tokens, Drafted: 10 tokens\n')
            value = metrics.summarize([{}], 2, log, 1)
            self.assertEqual(value['windows'], 2)
            self.assertAlmostEqual(value['acceptance_rate'], 10 / 12)
            self.assertEqual(value['source'], 'server_log')
            self.assertIn('日志观察值', metrics.label(value))
            self.assertIn('质量复核', value['scope'])
            self.assertEqual(metrics.summarize([], 0, log, 1)['status'], 'disabled')
            self.assertIsNone(metrics.summarize([], 0, log, 1)['acceptance_rate'])
            self.assertNotEqual(metrics.label(None), metrics.label(metrics.summarize([], 0, log, 1)))

    def test_partial_structured_rounds_do_not_fallback_or_average_missing(self):
        value = metrics.difference(snapshot(0, 0, 0), snapshot(2, 4, 3))
        for results in ([{'mtp_acceptance': value}], [{'mtp_acceptance': value}, {}]):
            self.assertEqual(metrics.summarize(results, 2, '/absent', 2)['status'], 'incomplete')

    def test_capture_preserves_raw_snapshot_and_handles_endpoint_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'metrics.prom'
            with patch.object(metrics, 'endpoint', return_value='not supported'):
                value = metrics.capture('http://test', path)
            self.assertEqual(path.read_text(), 'not supported')
            self.assertEqual(value['status'], 'unavailable')
            self.assertIn('sha256', value)
            with patch.object(metrics, 'endpoint', side_effect=TimeoutError('timeout')):
                self.assertEqual(metrics.capture('http://test', path)['status'], 'unavailable')

    def test_worker_collects_after_warmup_and_before_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            trial = make_trial(kernel='auto', mtp=2)
            phase = {'kind': 'final', 'candidates': [trial], 'samples': 1}
            audit = {'passed': True, 'methods': {}, 'kernels': {}, 'attention': {}, 'gdn': {}, 'activation_schemas': []}
            for name, value in {
                'config.json': {'server': {'port': 9999}, 'benchmark': {}},
                'active.json': {'phase': 'final', 'profile_id': trial['id']}, 'phases/final.json': phase,
                'model.json': {'activation_bits': 16}, 'inputs.json': [{'uid': 'p', 'group': 'g'}],
                'quality_inputs.json': [{'uid': 'q', 'group': 'g'}],
                f'trials/final/{trial["id"]}/audit.json': audit,
            }.items():
                write_json(run / name, value)
            events = []
            def request(base, row, settings):
                events.append(row['uid'])
                return {'status': 'success', 'ttft_s': .1, 'e2e_s': .2, 'decode_tps': 10}
            def capture(base, path):
                events.append(path.name)
                return snapshot(10, 20, 10) if 'before' in path.name else snapshot(12, 24, 13)
            arguments = ['worker', '--run-dir', str(run), '--result-dir', str(run / 'official'), '--result-filename', 'run=0.json']
            with patch('sys.argv', arguments), patch.object(worker, 'warmup', side_effect=lambda *args: events.append('warmup')), \
                    patch.object(worker, 'reset_cache'), patch.object(worker, 'request_one', side_effect=request), \
                    patch.object(worker, 'score', return_value={'score': 1}), patch.object(metrics, 'capture', side_effect=capture):
                worker.main()
            self.assertEqual(events, ['warmup', 'mtp-before.prom', 'p', 'p', 'mtp-after.prom', 'q', 'q'])
            result = read_json(run / 'official/run=0.json')
            self.assertEqual(result['mtp_acceptance']['acceptance_rate'], .75)
            self.assertEqual(result['cache']['warm']['quality_score'], 1)

    def test_terminal_progress_and_running_phase_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            trial = make_trial(kernel='auto')
            config = {'search': {'strategy': 'staged', 'objective_cache': 'warm', 'graphs': ['full'], 'budgets': [8192]}}
            plan = {'baseline': trial, 'capabilities': {'skipped': [{'option': 'marlin', 'reason': '所选W8A8不适配A16内核'}]}}
            stats = {'requests': 1, 'successes': 1, 'errors': 0, 'truncated': 0, 'e2e_s': .4, 'ttft_s': .1, 'decode_tps': 10, 'quality_score': 1}
            for name, value in {
                'candidates.json': {trial['id']: trial}, 'plan.json': plan,
                'phases/kernels.json': {'kind': 'screen', 'name': 'kernels', 'samples': 1, 'repeats': 1, 'candidates': [trial]},
                f'official/kernels/{trial["id"]}/run=0.json': {'stage2_status': 'complete', 'samples': 1, 'profile_id': trial['id'], 'cache': {'cold': stats, 'warm': stats}},
                'progress.json': {'phase': 'kernels', 'profile': trial['label'], 'round': 'run=0', 'completed': 1, 'total': 1},
            }.items():
                write_json(run / name, value)
            output = io.StringIO()
            process = Mock()
            process.wait.side_effect = [subprocess.TimeoutExpired('test', 0), 0]
            with patch('sys.stdout', output):
                self.assertEqual(wait_with_progress(process, run, 'kernels', interval=0), 0)
                publish_phase(run, config, {'precision': 'W8A8'}, 'kernels')
            self.assertIn('1/1', output.getvalue())
            self.assertIn('marlin', output.getvalue())
            self.assertIn('W8A8不适配', (run / 'search-details.md').read_text())
            html = (run / 'search-details.html').read_text()
            section = html.split('id="step-kernels"')[1].split('</section>')[0]
            self.assertIn('marlin：本轮跳过', section.replace('</strong>', ''))
            self.assertIn('尚无最终推荐', (run / 'search-details.md').read_text())


if __name__ == '__main__':
    unittest.main()
