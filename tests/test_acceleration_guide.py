"""Stage-two prompts, navigation, submitted budgets and dashboard use the same scope."""
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from pipeline import dashboard, jobs
from pipeline.common import write_json
from pipeline.workflow import build_configs, parse_arguments, resume_contract

ROOT = Path(__file__).resolve().parents[1]


class AccelerationScopeContracts(unittest.TestCase):
    def test_direct_acceleration_does_not_require_calibration_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(ROOT / 'configs', root / 'configs')
            (root / 'data').mkdir()
            for name in ('public100.json', 'quality100.json'):
                shutil.copy2(ROOT / 'data' / name, root / 'data' / name)
            args = parse_arguments(['--root', str(root), '--model', '/existing', '--skip-quantization'])
            quant, inference = build_configs(root, args)
            contract = resume_contract(root, args, quant, inference)
            self.assertEqual(set(contract['data_sha256']),
                             {str(root / 'data/public100.json'), str(root / 'data/quality100.json')})
            self.assertFalse((root / 'data/calibration-sources').exists())

    def test_submitted_and_frozen_scope_render_without_quantization(self):
        try:
            import rich
        except ImportError:
            self.skipTest('optional terminal dependencies not installed')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / 'jobs/direct/launch.json', {'attempt': 'test'})
            write_json(root / 'jobs/direct/state.json', {'attempt': 'test', 'state': 'completed'})
            write_json(root / 'jobs/direct/request.json', {'arguments': ['--skip-quantization']})
            data = dashboard.snapshot(root, 'direct')
            self.assertTrue(data['skip_quantization'])
            self.assertFalse(data['can_reselect'])
            write_json(root / 'runs/workflows/direct/request.json', {'skip_quantization': True})
            data = dashboard.snapshot(root, 'direct')
            for render in (dashboard.render, dashboard.render_pinned):
                display = render(data, {}, 'direct', width=120, height=30, colors=False)
                self.assertIn('已有模型', display)
                self.assertNotIn('量化与评测', display)
                self.assertNotIn('候选', display)
                self.assertNotIn('B 重选模型', display)
            write_json(root / 'runs/workflows/direct/request.json', {
                'skip_quantization': True, 'retest_source': {'candidate': 'awq4'}})
            self.assertTrue(dashboard.snapshot(root, 'direct')['can_reselect'])


class GuidedStageContracts(unittest.TestCase):
    def setUp(self):
        try:
            from pipeline import guided
        except ImportError:
            self.skipTest('optional terminal dependencies not installed')
        self.guided = guided
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        shutil.copytree(ROOT / 'configs', self.root / 'configs')
        self.resources = {'gpus': [{'index': 0}], 'storage': {}, 'memory': {}}
        self.model = {'path': '/existing', 'model_type': 'qwen3_5_moe', 'precision': 'bfloat16',
                      'weight_gib': 20.5, 'shards': 3, 'quantized': True,
                      'mtp_tensors': 19, 'max_model_len': 32768}

    def run_guide(self, skip=True, custom=False, back=False, invalid_budget=False):
        guided = self.guided
        self.model['quantized'] = skip
        prompts, headers, panels = [], [], []
        returned = False
        rejected_once = False
        def text(label, default='', **kwargs):
            nonlocal rejected_once
            prompts.append(label)
            if len(prompts) > 60:
                raise AssertionError("wizard failed to recover")
            if label == "加速初筛组合预算" and invalid_budget and not rejected_once:
                rejected_once = True
                return "1001"
            if label == '模型目录':
                return '/existing'
            if label == '任务名称':
                return 'direct'
            if '自定义测试' in label or 'MTP目录' in label:
                return ''
            return str({'加速初筛条数': 7, '加速初筛组合预算': 11, '加速最终复测轮数': 2}.get(label, default))
        def choose(title, options, **kwargs):
            nonlocal returned
            prompts.append(title)
            if title == '本次运行模式':
                return 'custom' if custom else 'standard'
            if title.startswith('量化方法'):
                return ['awq4']
            if title == '资源设置':
                return 'budget' if custom else 'keep'
            if title.startswith('量化模型推荐偏好'):
                return 'size'
            if '如何处理' in title or title == '加速完成后如何选择部署方案':
                return 'manual'
            if title == '下一步':
                if back and not returned:
                    returned = True
                    return 'resource'
                return 'start'
            raise AssertionError(title)
        with patch.object(guided.preflight, 'inspect_resources', return_value=self.resources), \
             patch.object(guided, 'resource_panel'), \
             patch.object(guided, 'model_input', return_value=Path('/existing')), \
             patch.object(guided.preflight, 'model_summary', return_value=self.model), \
             patch.object(guided.preflight, 'recipe_choices', return_value=[{'id':'awq4','label':'AWQ4','reason':None}]), \
             patch.object(guided.preflight, 'validate_resources', return_value={'errors': [], 'warnings': []}), \
             patch.object(guided.ui, 'heading', side_effect=lambda title, detail: headers.append((title, detail))), \
             patch.object(guided.ui, 'key_values', side_effect=lambda title, rows: panels.append((title, rows))), \
             patch.object(guided.ui, 'text_input', side_effect=text), \
             patch.object(guided.ui, 'choose', side_effect=choose), \
             patch.object(guided.ui.console, 'print'), \
             patch.object(guided.jobs, 'submit', return_value='direct') as submit:
            self.assertEqual(guided.guide(self.root, skip), 'direct')
        return prompts, headers, panels, submit.call_args.args[1]

    def test_existing_model_has_seven_real_steps_and_one_choice(self):
        prompts, headers, panels, args = self.run_guide(back=True)
        self.assertIn('--skip-quantization', args)
        for flag in ('--recipes', '--selection-objective', '--calibration-count', '--performance-repeats'):
            self.assertNotIn(flag, args)
        self.assertTrue(all('/7' in title for title, _ in headers))
        self.assertTrue(all('量化与评测' not in detail for _, detail in headers))
        self.assertNotIn('量化配方', str(headers))
        self.assertFalse(any('量化模型推荐偏好' in prompt for prompt in prompts))
        self.assertIn('加速完成后如何选择部署方案', prompts)
        self.assertTrue(any('配置 5/7 · 资源与预算' == title for title, _ in headers[-3:]))
        self.assertFalse(any(label == '校准' for _, rows in panels for label, _ in rows))
        self.assertNotIn('--auto-select', args)

    def test_custom_acceleration_budget_controls_actual_stage_two(self):
        prompts, _, panels, arguments = self.run_guide(custom=True)
        args = parse_arguments(['--root', str(self.root), *arguments])
        _, inference = build_configs(self.root, args)
        self.assertEqual(inference['search']['screening_samples'], 7)
        self.assertEqual(inference['search']['max_trials'], 11)
        self.assertEqual(inference['search']['repeats'], 2)
        self.assertFalse(any('校准' in prompt or '第一阶段' in prompt for prompt in prompts))
        final_rows = dict(panels[-1][1])
        self.assertIn('2轮', final_rows['加速最终复测'])
        self.assertIn('11个组合', final_rows['加速初筛'])

    def test_invalid_budget_can_be_corrected_and_preview_uses_local_config(self):
        config = json.loads((self.root / 'configs/inference.example.json').read_text())
        config['search'].update(repeats=5, max_trials=19)
        write_json(self.root / 'configs/inference.local.json', config)
        prompts, _, panels, arguments = self.run_guide(custom=True, invalid_budget=True)
        self.assertEqual(prompts.count('资源设置'), 2)
        first_budget = next(dict(rows) for name, rows in panels if name == '资源与测量设置')
        self.assertIn('5轮', first_budget['加速最终复测'])
        self.assertIn('19个组合', first_budget['加速初筛'])
        self.assertEqual(jobs.argument_value(arguments, '--search-trials'), '11')

    def test_new_model_still_has_quantization_choices(self):
        prompts, headers, _, args = self.run_guide(skip=False)
        self.assertNotIn('--skip-quantization', args)
        self.assertIn('--recipes', args)
        self.assertIn('--selection-objective', args)
        self.assertTrue(any('量化模型推荐偏好' in prompt for prompt in prompts))
        self.assertTrue(all('/8' in title for title, _ in headers))


if __name__ == '__main__':
    unittest.main()
