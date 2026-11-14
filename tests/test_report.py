import json
from pathlib import Path
import tempfile
import unittest

from pipeline.report import generate


class PublicReport(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {
            'quantization': {'method': 'awq4', 'source': '/models/Qwen3.5-4B'},
            'benchmark': {'concurrencies': [1, 4], 'cache_modes': ['cold', 'warm']},
            'profiles': [
                {'id': 'fa2', 'attention': 'fa2', 'graph': True},
                {'id': 'mtp', 'attention': 'flashinfer', 'graph': True, 'mtp': 2},
                {'id': 'fa3', 'attention': 'fa3', 'graph': True},
            ],
        }
        self.write('config.json', self.config)
        self.write('inputs.json', [{'uid': 'a'}, {'uid': 'b'}])
        self.write('model_manifest.json', {'mode': 'reused'})
        self.write('doctor.json', {'gpu': 'A800', 'vllm': '0.33.0'})
        self.write('profiles/fa3/status.json', {'status': 'skipped', 'reason': 'SM80'})
        for profile in ['fa2', 'mtp']:
            self.write('profiles/' + profile + '/status.json', {'status': 'completed'})
            self.write('profiles/' + profile + '/launch.json', {'command': [
                'python', '--quantization', 'awq_marlin', '--enable-prefix-caching',
                '--enable-chunked-prefill', '--max-num-batched-tokens', '8192', '--compilation-config',
                json.dumps({'mode': 3, 'cudagraph_mode': 'FULL_AND_PIECEWISE'})]})
            self.write('profiles/' + profile + '/loading_audit.json', {
                'passed': True, 'methods': {'AutoAWQMarlinLinearMethod': 1},
                'attention_implementations': {'Attention': 1},
                'compilation': 'CompilationMode.VLLM_COMPILE CUDAGraphMode.FULL_AND_PIECEWISE'})
            rows = []
            for concurrency in [1, 4]:
                for cache in ['cold', 'warm']:
                    # C4 and cold favour FA2; the requested C1/warm point favours MTP.
                    e2e = (2 if profile == 'mtp' else 5) if concurrency == 1 and cache == 'warm' else (7 if profile == 'mtp' else 3)
                    for index, uid in enumerate(['a', 'b']):
                        rows.append({'uid': uid, 'group': 'synthetic', 'status': 'success',
                                     'concurrency': concurrency, 'cache': cache,
                                     'e2e_s': e2e + index, 'ttft_s': .1 + .2 * index,
                                     'decode_tps': 100 + 20 * index, 'output_tokens': 10})
            path = self.root / 'profiles' / profile / 'requests.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_simple_report_uses_concurrency_one_and_warm_means(self):
        result = generate(self.root)
        self.assertEqual(result['summary']['concurrency'], 1)
        self.assertEqual(result['summary']['cache'], 'warm')
        self.assertEqual(result['recommendation']['profile'], 'mtp')
        context = result['summary']['acceleration_context']
        self.assertEqual({item['id'] for item in context['common_features']},
                         {'awq_marlin', 'compile', 'graph', 'kv_cache', 'prefix_cache', 'chunked_prefill'})
        self.assertEqual(context['comparison_dimensions'], ['注意力后端（FA2 / FlashInfer）', 'MTP（关闭 / MTP2）'])
        self.assertIn('分块预填充调度预算：8192 tokens', context['fixed_settings'])
        row = next(row for row in result['summary']['rows'] if row['profile'] == 'mtp' and row['cache'] == 'warm')
        self.assertAlmostEqual(row['ttft_s_mean'], .2)
        self.assertAlmostEqual(row['e2e_s_mean'], 2.5)
        self.assertAlmostEqual(row['decode_tps_mean'], 110)
        document = (self.root / 'report.html').read_text()
        self.assertIn('<th rowspan="2">加速方案</th>', document)
        self.assertIn('<th colspan="3" class="cache-group">cold：首次前缀</th>', document)
        self.assertIn('<th colspan="3" class="cache-group">warm：重复前缀</th>', document)
        self.assertEqual(document.count('<table>'), 1)
        self.assertEqual(document.count('<tr><td>FlashInfer + MTP2 + CUDA Graph</td>'), 1)
        self.assertIn('cold 推荐：FA2 + CUDA Graph', document)
        self.assertIn('warm 推荐：FlashInfer + MTP2 + CUDA Graph', document)
        self.assertIn('各方案共同开启：', document)
        self.assertIn('本次主要对比：', document)
        self.assertNotIn(' hidden', document)
        self.assertNotIn('<select', document)
        self.assertIn('FA3 + CUDA Graph：当前 GPU 不支持，已跳过', document)
        for removed in ['P95', 'QPM', '分桶', '<svg', '<pre']:
            self.assertNotIn(removed, document)
        self.assertFalse((self.root / 'report-detailed.html').exists())

    def test_details_are_explicit_and_do_not_replace_simple_report(self):
        generate(self.root, detailed=True)
        self.assertIn('P95', (self.root / 'report-detailed.html').read_text())
        self.assertNotIn('P95', (self.root / 'report.html').read_text())

    def test_incomplete_cohort_is_not_recommended(self):
        path = self.root / 'profiles/mtp/requests.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [row for row in rows if not (row['concurrency'] == 1 and row['cache'] == 'warm' and row['uid'] == 'b')]
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        result = generate(self.root)
        self.assertIsNone(result['recommendation']['profile'])
        incomplete = next(row for row in result['summary']['rows'] if row['profile'] == 'mtp' and row['cache'] == 'warm')
        self.assertFalse(incomplete['complete'])
        self.assertIsNone(incomplete['e2e_s_mean'])

    def test_missing_concurrency_one_does_not_fall_back_to_four(self):
        self.config['benchmark']['concurrencies'] = [4]
        self.write('config.json', self.config)
        result = generate(self.root)
        self.assertEqual(result['summary']['rows'], [])
        self.assertIsNone(result['recommendation']['profile'])

    def test_disabled_features_are_not_advertised_as_common(self):
        self.write('profiles/mtp/launch.json', {'command': [
            'python', '--quantization', 'awq_marlin', '--no-enable-prefix-caching',
            '--enable-chunked-prefill', '--enforce-eager']})
        result = generate(self.root)
        features = {item['id'] for item in result['summary']['acceleration_context']['common_features']}
        self.assertEqual(features, {'awq_marlin', 'kv_cache', 'chunked_prefill'})

    def test_missing_launch_record_does_not_infer_shared_features(self):
        (self.root / 'profiles/mtp/launch.json').unlink()
        result = generate(self.root)
        self.assertEqual(result['summary']['acceleration_context']['common_features'], [])


if __name__ == '__main__':
    unittest.main()
