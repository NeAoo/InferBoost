import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from pipeline.inference import assets, planning
from pipeline.inference.cli import resolve_model
from pipeline.inference.execution import complete, phase_results
from pipeline.inference.report import generate
from pipeline.inference.deploy import select
from pipeline.inference.worker import execution_signature, representative_rows
from pipeline.common import digest


class InferenceContracts(unittest.TestCase):
    def test_native_int8_qkv_is_not_a_protected_bf16_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root/'checkpoint'; model.mkdir()
            run = root/'run'; run.mkdir()
            config = {'model_type':'qwen3_5', 'text_config':{}, 'quantization_config':{
                'quant_method':'compressed-tensors','ignore':[], 'config_groups':{
                    'g':{'weights':{'num_bits':8},'input_activations':{'num_bits':8,'type':'int'}}}}}
            (model/'config.json').write_text(json.dumps(config))
            prefix = 'model.language_model.layers.0.linear_attn.'
            header = {prefix+'in_proj_a.weight':{'dtype':'BF16','shape':[2],'data_offsets':[0,4]},
                      prefix+'in_proj_b.weight':{'dtype':'BF16','shape':[2],'data_offsets':[4,8]},
                      prefix+'in_proj_qkv.weight':{'dtype':'I8','shape':[2],'data_offsets':[8,10]}}
            raw = json.dumps(header).encode()
            (model/'model.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+bytes(10))
            def prepare_references(*args, **kwargs):
                (run/'assets').mkdir()
                (run/'assets/expected_bf16.json').write_text('{}')
            with patch('pipeline.inference.assets.subprocess.run',side_effect=prepare_references):
                result = assets.prepare(root,run,{'runtime':{'python':'/python'}},model)
            ignored = json.loads((run/'model/config.json').read_text())['quantization_config']['ignore']
            self.assertNotIn(prefix+'in_proj_qkv',ignored)
            self.assertIn(prefix+'in_proj_a',ignored)
            self.assertIn(prefix+'in_proj_ba',ignored)
            self.assertEqual(len(result['protected_modules']),2)

    def setUp(self):
        self.config = {'runtime': {'python': '/opt/venv/bin/python'},
                       'server': {'port': 18550, 'max_model_len': 32768, 'max_num_seqs': 4,
                                  'gpu_memory_utilization': .9},
                       'search': {'attention': ['fa2', 'fa3', 'flashinfer'],
                                  'kernels': ['auto', 'marlin', 'humming'],
                                  'graphs': ['none', 'piecewise', 'full'], 'budgets': [4096, 8192],
                                  'mtp_depths': [0, 1, 2], 'max_trials': 24, 'repeats': 2,
                                  'objective_cache': 'warm'}}
        self.model = {'method': 'awq', 'bits': 4, 'activation_bits': 16, 'activation_type': 'int',
                      'load_view': '/models/view', 'model_type': 'qwen3_5', 'mtp_supported': False}
        self.hardware = {'capability': [8, 0], 'flashinfer': True, 'humming': False}

    def test_hardware_and_precision_filtering(self):
        result = planning.capabilities(self.model, self.hardware, self.config)
        self.assertEqual(result['allowed']['attention'], ['fa2', 'flashinfer'])
        self.assertNotIn('humming', result['allowed']['kernels'])
        self.assertEqual(result['allowed']['mtp_depths'], [0])
        model = dict(self.model, activation_bits=8, activation_type='float')
        with self.assertRaises(ValueError):
            planning.capabilities(model, self.hardware, self.config)
        result = planning.capabilities(model, dict(self.hardware, capability=[9, 0]), self.config)
        self.assertNotIn('marlin', result['allowed']['kernels'])

    def test_compile_and_graph_are_independent(self):
        compiled = planning.make_trial(compile=True, graph='none')
        command = planning.server_command(self.config, self.model, compiled)
        self.assertNotIn('--enforce-eager', command)
        self.assertEqual(json.loads(command[command.index('--compilation-config') + 1]),
                         {'mode': 3, 'cudagraph_mode': 'NONE'})
        eager = planning.server_command(self.config, self.model, planning.make_trial(compile=False, graph='none'))
        self.assertIn('--enforce-eager', eager)
        self.assertNotIn('--compilation-config', eager)
        self.assertEqual(command[command.index('--kv-cache-dtype') + 1], 'auto')

    def test_full_search_is_bounded_and_mtp_requires_assets(self):
        capability = planning.capabilities(self.model, self.hardware, self.config)
        self.config['search']['max_trials'] = 1
        with self.assertRaises(ValueError):
            planning.full_grid(capability, self.config)
        with self.assertRaises(ValueError):
            planning.server_command(self.config, self.model, planning.make_trial(mtp=2))

    def test_screening_preserves_groups_and_has_no_duplicates(self):
        rows = [{'uid': str(g) + '-' + str(i), 'group': str(g)} for g in range(5) for i in range(20)]
        selected = representative_rows(rows, 20)
        self.assertEqual(len({row['uid'] for row in selected}), 20)
        self.assertEqual({row['group'] for row in selected}, {str(i) for i in range(5)})
        self.assertTrue(any(row['uid'].endswith('-19') for row in selected))

    def test_dedup_uses_actual_kernels_not_requested_names(self):
        audit = {'methods': {'AWQ': 1}, 'kernels': {'MarlinLinearKernel': 1},
                 'attention': {'FA': 1}, 'gdn': {}, 'activation_schemas': []}
        first = planning.make_trial(kernel='auto')
        second = planning.make_trial(kernel='marlin')
        self.assertEqual(execution_signature(audit, first), execution_signature(audit, second))
        changed = dict(audit, kernels={'HummingLinearKernel': 1})
        self.assertNotEqual(execution_signature(audit, first), execution_signature(changed, second))

    def test_model_inspection_and_partial_mtp_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {'model_type': 'qwen3_5', 'text_config': {'hidden_size': 2},
                      'quantization_config': {'quant_method': 'compressed-tensors', 'config_groups': {
                          'g': {'weights': {'num_bits': 4}, 'input_activations': {'num_bits': 8, 'type': 'float'}}}}}
            (root / 'config.json').write_text(json.dumps(config))
            header = {'mtp.fc.weight': {'dtype': 'BF16', 'shape': [2, 4], 'data_offsets': [0, 16]},
                      'other': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [16, 20]}}
            raw = json.dumps(header).encode()
            (root / 'model.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(20))
            model = assets.inspect(root)
            self.assertEqual(model['precision'], 'W4A8')
            self.assertEqual(model['activation_type'], 'float')
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'mtp.fc.weight': 'model.safetensors'}}))
            self.assertTrue(assets.inspect_mtp(root)['bf16_mtp_present'])

    def test_stage_one_handoff_checks_user_choice_and_loading_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            view = root / 'model'; view.mkdir(); (view / 'config.json').write_text('{}')
            candidate = root / 'candidates/a'; candidate.mkdir(parents=True)
            (candidate / 'model_manifest.json').write_text(json.dumps({'load_config_sha256': digest(view / 'config.json')}))
            handoff = {'schema_version': 1, 'stage': 'quantization_selection', 'candidate': 'a',
                       'model': str(view), 'quality_gate': {'passed': True}}
            selected = root / 'selected_model.json'; selected.write_text(json.dumps(handoff))
            self.assertEqual(resolve_model(selected)[0], view.resolve())
            handoff['quality_gate']['passed'] = False
            selected.write_text(json.dumps(handoff))
            with self.assertRaises(ValueError):
                resolve_model(selected)
            handoff['selection'] = {'mode': 'user', 'quality_criteria_are_advisory': True}
            selected.write_text(json.dumps(handoff))
            self.assertEqual(resolve_model(selected)[0], view.resolve())
            (view / 'config.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                resolve_model(selected)

    def test_changed_loading_view_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'config.json'
            path.write_text('{}')
            model = {'loading_files_sha256': {str(path): digest(path)}}
            assets.verify_loading_files(model)
            path.write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                assets.verify_loading_files(model)

    def test_report_rejects_missing_rounds_and_handles_single_token_tps(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            def write(name, value):
                path = run / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
            trials = [planning.make_trial(), planning.make_trial(attention='flashinfer')]
            write('config.json', self.config)
            write('candidates.json', {trial['id']: trial for trial in trials})
            write('finalists.json', {'baseline_id': trials[0]['id'], 'profiles': [trial['id'] for trial in trials]})
            write('model.json', {'path': '/models/selected', 'precision': 'W4A16'})
            write('hardware.json', {'gpu': 'A800'})
            write('inputs.json', [{'uid': 'a'}, {'uid': 'b'}])
            write('plan.json', {'capabilities': {'skipped': []}})
            stats = {'requests': 2, 'successes': 2, 'errors': 0, 'truncated': 0,
                     'ttft_s': .1, 'e2e_s': .2, 'decode_tps': None, 'quality_score': None}
            result = {'stage2_status': 'complete', 'samples': 2, 'cache': {'cold': stats, 'warm': stats}}
            self.assertTrue(complete(result, 2))
            for index in range(2):
                write('official/final/' + trials[0]['id'] + '/run=' + str(index) + '.json', dict(result, profile_id=trials[0]['id']))
            write('official/final/' + trials[1]['id'] + '/run=0.json', dict(result, profile_id=trials[1]['id']))
            write('official/final/' + trials[1]['id'] + '/run=1.failure.json', {'error': 'server stopped'})
            write('official/final/SERVE-bad-failure.json', {'serve_overrides': {'profile_id': 'bad'}, 'error': 'startup failed'})
            screened = planning.make_trial(mtp=2)
            write('phases/mtp.json', {'name': 'mtp', 'kind': 'screen', 'samples': 1, 'repeats': 1,
                                     'candidates': [screened]})
            screen_stats = dict(stats, requests=1, successes=1, e2e_s=.001)
            write('official/mtp/' + screened['id'] + '/run=0.json',
                  {'stage2_status': 'complete', 'samples': 1, 'profile_id': screened['id'],
                   'cache': {'cold': screen_stats, 'warm': screen_stats}})
            self.assertEqual(len(phase_results(run, 'final')), 3)
            report = generate(run)
            self.assertEqual(report['recommendation'], trials[0]['id'])
            self.assertEqual(len(report['excluded']), 1)
            self.assertIsNone(report['rows'][0]['decode_tps_mean'])
            self.assertEqual(report['failed_candidates'], ['bad'])
            self.assertEqual(report['search_coverage']['complete_screened_candidates'], 1)
            self.assertNotIn(screened['id'], [row['profile'] for row in report['rows']])
            evidence = json.loads((run / 'search-details.json').read_text())
            self.assertEqual(evidence['records'][0]['samples_per_cache'], 1)
            self.assertEqual(evidence['records'][0]['results'][0]['cache']['warm']['e2e_s'], .001)
            self.assertIsNone(evidence['records'][0]['observed_kernels'])
            document = (run / 'report.html').read_text()
            self.assertEqual(document.count('<table>'), 1)
            self.assertNotIn('<svg', document)
            self.assertIn(trials[0]['label'], (run / 'report.md').read_text())
            self.assertIn('不把完整网格或固定配置验证描述为五步筛选', (run / 'search-details.md').read_text())

    def test_request_only_custom_data_can_recommend_without_claiming_accuracy(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            def write(name, value):
                path = run / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            trial = planning.make_trial()
            write('config.json', dict(self.config, search=dict(self.config['search'], repeats=1),
                                      quality={'dataset': '/custom.json'}))
            write('candidates.json', {trial['id']: trial})
            write('finalists.json', {'baseline_id': trial['id'], 'profiles': [trial['id']]})
            write('model.json', {'path': '/models/selected', 'precision': 'W8A8'})
            write('hardware.json', {'gpu': 'A800'})
            write('inputs.json', [{'uid': 'a'}])
            write('quality_inputs.json', [{'uid': 'a'}])
            write('plan.json', {'capabilities': {'skipped': []}})
            stats = {'requests': 1, 'successes': 1, 'errors': 0, 'truncated': 0,
                     'ttft_s': .1, 'e2e_s': .2, 'decode_tps': 10,
                     'quality_score': None, 'quality_scored_count': 0, 'quality_expected_count': 1}
            write('official/final/' + trial['id'] + '/run=0.json',
                  {'stage2_status': 'complete', 'profile_id': trial['id'], 'samples': 1,
                   'cache': {'cold': stats, 'warm': stats}})
            report = generate(run)
            self.assertEqual(report['recommendation'], trial['id'])
            self.assertTrue(report['quality_comparison_ready'])
            self.assertFalse(report['quality_scored'])
            self.assertIsNone(report['rows'][0]['quality_score'])
            self.assertIn('未提供完整标准答案时不计算准确率', (run / 'report.html').read_text())
            write('quality_inputs.json', [{'uid': 'a', 'expected': 1, 'scorer': 'number'}])
            report = generate(run)
            self.assertIsNone(report['recommendation'])
            self.assertFalse(report['quality_comparison_ready'])

    def test_recommendation_requires_quality_in_both_cache_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            def write(name, value):
                path = run / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
            trials = [planning.make_trial(), planning.make_trial(mtp=1)]
            write('config.json', dict(self.config, search=dict(self.config['search'], repeats=1)))
            write('candidates.json', {trial['id']: trial for trial in trials})
            write('finalists.json', {'baseline_id': trials[0]['id'], 'profiles': [trial['id'] for trial in trials]})
            write('model.json', {'path': '/models/selected', 'precision': 'W8A16'})
            write('hardware.json', {'gpu': 'A800'})
            write('inputs.json', [{'uid': 'a'}])
            write('plan.json', {'capabilities': {'skipped': []}})
            stats = {'requests': 1, 'successes': 1, 'errors': 0, 'truncated': 0,
                     'ttft_s': .1, 'e2e_s': .2, 'decode_tps': 10, 'quality_score': 1}
            for index, trial in enumerate(trials):
                result = {'stage2_status': 'complete', 'profile_id': trial['id'], 'samples': 1,
                          'cache': {'cold': stats, 'warm': stats}}
                if index:
                    result['cache'] = {'cold': dict(stats, quality_score=0), 'warm': dict(stats, e2e_s=.1)}
                write('official/final/' + trial['id'] + '/run=0.json', result)
            report = generate(run)
            self.assertEqual(report['best_by_cache']['warm'], trials[0]['id'])
            self.assertFalse(report['quality_gates'][trials[1]['id']])
            self.assertIn('质量复核未达标', (run / 'report.html').read_text())
            self.assertEqual(report['report_generator_sha256'], digest(Path('pipeline/inference/report.py')))
            write('official/final/' + trials[1]['id'] + '/run=0.json',
                  {'stage2_status': 'complete', 'profile_id': trials[1]['id'], 'samples': 1,
                   'cache': {'cold': stats, 'warm': dict(stats, e2e_s=.199)}})
            report = generate(run)
            self.assertEqual(report['recommendation'], trials[1]['id'])
            self.assertIn('差距较小', (run / 'report.html').read_text())
            write('config.json', dict(self.config, search=dict(self.config['search'], repeats=1),
                                      quality={'dataset': '/quality100.json'}))
            (run / ('official/final/' + trials[0]['id'] + '/run=0.json')).unlink()
            report = generate(run)
            self.assertIsNone(report['recommendation'])
            self.assertFalse(report['quality_comparison_ready'])
            with self.assertRaises(ValueError):
                select(run, trials[1]['id'])


if __name__ == '__main__':
    unittest.main()
