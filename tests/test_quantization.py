import json
import copy
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from pipeline import quality
from pipeline.quantization import inference_config, tensor_fingerprints, interactive, select, load_config
from pipeline.quantization_catalog import FRESH_RECIPES, RECIPES, eligibility, ignore_patterns, inspect_source
from pipeline.quantization_report import generate, resource_metrics
from pipeline.common import digest
from pipeline.runtime import build_command
from scripts.quantize_compressor import validate_main_model_only


class QualityContracts(unittest.TestCase):
    def test_unlabeled_data_is_not_accuracy(self):
        row = {'uid': 'a', 'group': 'x', 'messages': [{'role': 'user', 'content': 'question'}]}
        result = quality.summarize([row], [{'uid': 'a', 'status': 'success', 'content': 'anything'}])
        self.assertIsNone(result['score_pct'])
        self.assertEqual(result['unscored_count'], 1)

    def test_number_and_json_format_are_strict(self):
        response = lambda answer: {'status': 'success', 'content': answer}
        numeric = {'expected': 12, 'scorer': 'number'}
        self.assertEqual(quality.score(numeric, response('12.0'))['score'], 1)
        for bad in ['NaN', 'Infinity', '12件', '答案12']:
            self.assertEqual(quality.score(numeric, response(bad))['score'], 0)
        row = {'expected': {'count': 1}, 'scorer': 'json_exact'}
        self.assertEqual(quality.score(row, response('{"count": true}'))['score'], 0)
        self.assertEqual(quality.score(row, response('```json\n{"count": 1}\n```'))['score'], 0)
        row['allow_code_fence'] = True
        self.assertEqual(quality.score(row, response('```json\n{"count": 1}\n```'))['score'], 1)

    def test_wrong_cohort_cannot_pass_completeness(self):
        row = {'uid': 'a', 'group': 'x', 'expected': 'A', 'scorer': 'exact'}
        result = quality.summarize([row], [{'uid': 'b', 'status': 'success', 'content': 'A'}])
        self.assertFalse(result['complete'])
        self.assertIsNone(result['score_pct'])

    def test_error_or_truncation_is_not_correct_answer(self):
        for status in ['error', 'truncated']:
            self.assertEqual(quality.score({'expected': 'A', 'scorer': 'exact'},
                                           {'status': status, 'content': 'A'})['score'], 0)


class BackendContracts(unittest.TestCase):
    def test_default_gate_and_explicit_strict_gate(self):
        project = Path(__file__).resolve().parents[1]
        config = json.loads((project / 'configs/quantization.example.json').read_text())
        self.assertEqual(config['quality_gate'], {'max_drop_pp': 5.0, 'max_regressions': 3})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'config.json'
            del config['quality_gate']
            path.write_text(json.dumps(config))
            self.assertEqual(load_config(project, path)['quality_gate'],
                             {'max_drop_pp': 5.0, 'max_regressions': 3})
            for gate in ({'max_drop_pp': 0, 'max_regressions': 0}, {'max_drop_pp': 0}):
                config['quality_gate'] = gate
                path.write_text(json.dumps(config))
                self.assertEqual(load_config(project, path)['quality_gate'], gate)
            for gate in (None, {'max_drop_pp': float('nan')}, {'max_drop_pp': '5'},
                         {'max_drop_pp': -1}, {'max_regressions': 3.5}, {'max_regressions': True}):
                config['quality_gate'] = gate
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    load_config(project, path)

    def test_new_quantization_uses_compressor_and_old_exports_remain_reusable(self):
        self.assertTrue(FRESH_RECIPES)
        self.assertTrue(all(RECIPES[name].backend == 'llmcompressor' for name in FRESH_RECIPES))
        project = Path(__file__).resolve().parents[1]
        config = json.loads((project / 'configs/quantization.example.json').read_text())
        config['candidates'] = [{'id': 'old', 'recipe': 'gptq4', 'mode': 'fresh'}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'config.json'
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, '新量化已统一LLM Compressor'):
                load_config(project, path)
            config['candidates'][0].update(mode='reuse', model='/historical/export')
            path.write_text(json.dumps(config))
            loaded = load_config(project, path)
            self.assertEqual(loaded['candidates'][1]['mode'], 'reuse')

    def test_qwen_release_names_reuse_the_verified_architecture_identifiers(self):
        project = Path(__file__).resolve().parents[1]
        snapshots = [('qwen3.6-35b-a3b', 'qwen35_moe'), ('qwen3.8-27b', 'qwen35_dense'),
                     ('qwen3.8-2.4t-a95b', 'qwen35_moe')]
        for filename, family in snapshots:
            config = json.loads((project / f'references/model-configs/{filename}.json').read_text())
            with patch('pipeline.quantization_catalog.inspect_model', return_value=(
                    {'config': config, 'shards': {}}, {})):
                inspected = inspect_source('/configuration_snapshot_only')
            self.assertEqual(inspected['adapter']['family'], family)
            self.assertTrue(eligibility('compressor_gptq4', inspected, [8, 0])['eligible'])
            if family == 'qwen35_moe':
                self.assertTrue(eligibility('compressor_awq4', inspected, [8, 0])['eligible'])

    def test_unknown_architecture_has_no_implicit_awq_mapping(self):
        with patch('pipeline.quantization_catalog.inspect_model', return_value=(
                {'config': {'model_type': 'new_unsupported_model'}}, {})):
            with self.assertRaisesRegex(ValueError, '尚无经过审查的模型适配器'):
                inspect_source('/unused')

    def test_mtp_loader_guard_and_structurally_checked_moe_awq(self):
        class Model:
            def __init__(self, names):
                self.names = names
            def named_modules(self):
                return [(name, object()) for name in self.names]
        validate_main_model_only(Model(['', 'model.layers.0', 'model.layers.0.mlp']))
        for name in ['mtp', 'mtp.layers.0', 'model.mtp_layer']:
            with self.assertRaisesRegex(ValueError, '包含MTP模块'):
                validate_main_model_only(Model(['', name]))
        model = {'model_type': 'qwen3_5_moe', 'adapter': {'family': 'qwen35_moe', 'moe': True}}
        self.assertTrue(eligibility('compressor_awq4', model, [9, 0])['eligible'])
        self.assertTrue(eligibility('compressor_awq_gptq4', model, [9, 0])['eligible'])
        self.assertTrue(eligibility('gptq4', model, [8, 0])['eligible'])

    def test_precision_does_not_quantize_mtp_and_sm80_skips_w4afp8(self):
        model = {'model_type': 'qwen3_5', 'adapter': {'family': 'qwen35_dense', 'moe': False}}
        self.assertFalse(eligibility('compressor_w4afp8', model, [8, 0])['eligible'])
        self.assertTrue(eligibility('compressor_w4afp8', model, [9, 0])['eligible'])
        self.assertEqual(eligibility('compressor_w4afp8', model)['mtp_policy'], 'preserve_original_precision')
        self.assertTrue(any('mtp' in pattern for pattern in ignore_patterns(model)))

    def test_reference_has_no_quantization_or_mtp_flag(self):
        config = {'runtime': {'python': '/python'}, 'server': {'max_model_len': 1024, 'max_num_seqs': 1}}
        execution, profile = inference_config(config, {'id': 'original', 'recipe': 'original'})
        command = build_command(execution, profile, '/model', 12345)
        self.assertNotIn('--quantization', command)
        self.assertNotIn('--speculative-config', command)
        quantized, profile = inference_config(config, {'id': 'awq', 'recipe': 'awq4'})
        command = build_command(quantized, profile, '/model', 12345)
        self.assertEqual(command[command.index('--quantization') + 1], 'awq_marlin')

    def test_new_main_model_does_not_inherit_previous_mtp_source(self):
        config = {'source': '/models/old', 'mtp_source': '/models/old',
                  'candidates': [{'id': 'original', 'recipe': 'original'},
                                 {'id': 'gptq4', 'recipe': 'gptq4', 'mode': 'reuse', 'model': '/old/export'}],
                  'datasets': [{'id': 'default', 'path': '/data/tests.json'}]}
        model = {'model_type': 'qwen3_5', 'adapter': {'family': 'qwen35_dense', 'moe': False}}
        with patch('pipeline.quantization.load_config', return_value=copy.deepcopy(config)), \
             patch('pipeline.quantization.inspect_source', return_value=model), \
             patch('builtins.input', side_effect=['', '/models/new', 'all', '', '']), \
             patch('builtins.print'), \
             patch('pipeline.quantization.write_json') as write, \
             patch('pipeline.quantization.run', return_value=0):
            interactive('/unused')
        saved = write.call_args.args[1]
        self.assertEqual(saved['source'], '/models/new')
        self.assertEqual(saved['mtp_source'], '/models/new')
        self.assertTrue(all(row['mode'] == 'fresh' for row in saved['candidates'][1:]))

    def test_mtp_payload_change_detected_even_with_identical_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key = 'mtp.fc.weight'
            header = json.dumps({key: {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]}}).encode()
            path = root / 'model.safetensors'
            path.write_bytes(struct.pack('<Q', len(header)) + header + b'1234')
            before = tensor_fingerprints(root, [key])
            path.write_bytes(struct.pack('<Q', len(header)) + header + b'1235')
            self.assertNotEqual(before, tensor_fingerprints(root, [key]))
            with self.assertRaises(ValueError):
                tensor_fingerprints(root, ['mtp.missing.weight'])


class ComparisonContracts(unittest.TestCase):
    def test_tolerated_regressions_require_both_limits_and_complete_success(self):
        cases = [
            # count, original wrong, candidate wrong, request status, candidate status, pass
            (100, 0, 3, 'success', 'completed', True),
            (100, 0, 4, 'success', 'completed', False),
            (20, 0, 1, 'success', 'completed', True),   # exactly 5 percentage points
            (19, 0, 1, 'success', 'completed', False),  # >5 points, only one regression
            (100, 5, 4, 'success', 'completed', False), # score improves, but 4 new wrong
            (100, 0, 0, 'error', 'completed', False),
            (100, 0, 0, 'missing', 'completed', False),
            (100, 0, 0, 'success', 'failed', False),
        ]
        for count, original_wrong, candidate_wrong, request_status, candidate_status, expected in cases:
            with self.subTest(case=(count, original_wrong, candidate_wrong, request_status, candidate_status)), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                def write(name, value):
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(value))
                gate = {'max_drop_pp': 5.0, 'max_regressions': 3}
                write('config.json', {'candidates': [{'id': 'original', 'recipe': 'original'},
                                                    {'id': 'fast', 'recipe': 'awq4'}],
                                     'datasets': [{'id': 'default'}], 'quality_gate': gate})
                rows = [{'uid': str(i), 'group': 'test', 'expected': 'A', 'scorer': 'exact'} for i in range(count)]
                write('datasets/default.json', rows)
                for name in ['original', 'fast']:
                    write(f'candidates/{name}/status.json', {'status': candidate_status if name == 'fast' else 'completed'})
                    path = root / f'candidates/{name}/datasets/default/requests.jsonl'
                    path.parent.mkdir(parents=True)
                    records = []
                    for i, row in enumerate(rows):
                        wrong = i >= count - original_wrong if name == 'original' else i < candidate_wrong
                        status = request_status if name == 'fast' and i == 0 else 'success'
                        if status != 'missing':
                            records.append(dict(uid=row['uid'], group='test', content='B' if wrong else 'A',
                                                concurrency=1, cache='cold', status=status,
                                                e2e_s=1 if name == 'original' else .5))
                    path.write_text(''.join(json.dumps(row) + '\n' for row in records))
                report = generate(root)
                self.assertEqual(report['candidate_gates']['fast']['passed'], expected)
                self.assertEqual(report['recommendation']['candidate'], 'fast' if expected else 'original')
                self.assertEqual(report['quality_gate'], gate)
                self.assertIn('总分下降 ≤ 5 分', (root / 'report.md').read_text())
                self.assertIn('量化后答错 ≤ 3 道', (root / 'report.html').read_text())

    def test_resources_distinguish_model_memory_cache_and_cpu_ram(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'server.log').write_text('Model loading took 4.75 GiB memory and 12 seconds\nAvailable KV cache memory: 65.23 GiB\n')
            samples = [{'returncode': 0, 'sample': '2026/11/10 00:00:00, 50, 4096, 100, 1000, 1000'},
                       {'returncode': 0, 'sample': '2026/11/10 00:00:05, 50, N/A, 100, 1000, 1000'},
                       {'returncode': 1, 'sample': '2026/11/10 00:00:10, 50, 999999, 100, 1000, 1000'}]
            (root / 'gpu.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in samples))
            metrics = resource_metrics(root, {})
            self.assertEqual(metrics['model_load_gpu_gib'], 4.75)
            self.assertEqual(metrics['service_gpu_sample_peak_gib'], 4)
            self.assertEqual(metrics['kv_cache_budget_gib'], 65.23)
            self.assertIsNone(metrics['quantization_cpu_peak_gib'])

    def test_user_can_select_a_degraded_complete_candidate_but_not_a_failed_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'runs/manual'
            def write(name, value):
                path = run / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            write('config.json', {'source': '/original', 'mtp_policy': 'preserve_original_precision',
                  'candidates': [{'id': 'original', 'recipe': 'original'}, {'id': 'fast', 'recipe': 'awq4'}],
                  'datasets': [{'id': 'default'}], 'quality_gate': {'max_drop_pp': 0, 'max_regressions': 0}})
            write('datasets/default.json', [{'uid': 'a', 'group': 'x', 'expected': 'A', 'scorer': 'exact'}])
            for name, answer, seconds in [('original', 'A', 2), ('fast', 'B', .1)]:
                write(f'candidates/{name}/status.json', {'status': 'completed'})
                path = run / f'candidates/{name}/datasets/default/requests.jsonl'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'uid': 'a', 'group': 'x', 'content': answer, 'status': 'success',
                                            'concurrency': 1, 'cache': 'cold', 'e2e_s': seconds}) + '\n')
            write('candidates/fast/model/config.json', {})
            view = run / 'candidates/fast/model'
            write('candidates/fast/model_manifest.json', {'artifact': str(view), 'load_view': str(view),
                  'identity': {}, 'load_config_sha256': digest(view / 'config.json'), 'mtp_unchanged': True})
            write('candidates/fast/loading_audit.json', {'passed': True})
            write('calibration_manifest.json', {'version': 'pipeline-official-calibration-v1',
                                               'source_dataset': 'mlabonne/open-perfectblend'})
            with patch('pipeline.quantization.models.inspect_model', return_value=({}, {})), patch('builtins.print'):
                handoff = select(root, 'manual', 'fast')
            self.assertFalse(handoff['quality_gate']['passed'])
            self.assertEqual(handoff['selection']['mode'], 'user')
            self.assertEqual(handoff['selection']['recommended_candidate'], 'original')
            self.assertEqual(handoff['calibration']['source_dataset'], 'mlabonne/open-perfectblend')
            self.assertEqual(handoff['quantization']['recipe'], 'awq4')
            from pipeline.inference.cli import resolve_model
            self.assertEqual(resolve_model(run / 'selected_model.json')[0], view.resolve())
            write('candidates/fast/status.json', {'status': 'failed'})
            with self.assertRaisesRegex(ValueError, '完整测试'):
                select(root, 'manual', 'fast')

    def test_unrun_candidate_has_unknown_score_and_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'datasets').mkdir()
            (root / 'config.json').write_text(json.dumps({
                'candidates': [{'id': 'original', 'recipe': 'original'}, {'id': 'skipped', 'recipe': 'awq4'}],
                'datasets': [{'id': 'default'}], 'quality_gate': {'max_drop_pp': 0, 'max_regressions': 0}}))
            (root / 'datasets/default.json').write_text(json.dumps([
                {'uid': 'a', 'group': 'x', 'expected': 'A', 'scorer': 'exact'}]))
            report = generate(root)
            for row in report['overview']:
                self.assertIsNone(row['score_pct'])
                self.assertIsNone(row['quality_gate_passed'])
            self.assertIsNone(report['recommendation']['candidate'])
            self.assertIn('无法判断是否新增退化题', (root / 'report.html').read_text())
            self.assertEqual(report['progress_state'], 'in_progress')
            self.assertIn('进行中，暂定建议', (root / 'report.md').read_text())
            self.assertIn('规则题得分', (root / 'report.md').read_text())

    def test_faster_but_degraded_candidate_is_not_recommended(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def write(path, value):
                path = root / path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            config = {'candidates': [{'id': 'original', 'recipe': 'original'}, {'id': 'fast', 'recipe': 'awq4'}],
                      'datasets': [{'id': 'default'}], 'quality_gate': {'max_drop_pp': 0, 'max_regressions': 0}}
            write('config.json', config)
            write('datasets/default.json', [{'uid': 'a', 'group': 'x', 'expected': 'A', 'scorer': 'exact'}])
            for name, answer, seconds in [('original', 'A', 2), ('fast', 'B', .1)]:
                write(f'candidates/{name}/status.json', {'status': 'completed'})
                path = root / f'candidates/{name}/datasets/default/requests.jsonl'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'uid': 'a', 'group': 'x', 'content': answer, 'status': 'success',
                                            'concurrency': 1, 'cache': 'cold', 'e2e_s': seconds, 'ttft_s': .01, 'decode_tps': 100}) + '\n')
            report = generate(root)
            self.assertEqual(report['recommendation']['candidate'], 'original')
            self.assertFalse(report['candidate_gates']['fast']['passed'])
            self.assertEqual(report['candidate_gates']['fast']['datasets'][0]['regressions'], ['a'])


if __name__ == '__main__':
    unittest.main()
