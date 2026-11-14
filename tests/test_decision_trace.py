import copy
import unittest

from pipeline.inference.decision_trace import build
from pipeline.inference.decision_view import render
from pipeline.inference.planning import make_trial
from pipeline.inference.text_report import stage_markdown


def measured(phase, seconds, *, quality=1, complete=True, samples=20, **settings):
    trial = make_trial(kernel='humming', attention='flashinfer', **settings)
    stats = {'e2e_s': seconds, 'ttft_s': .1, 'decode_tps': 50, 'quality_score': quality}
    return {'phase': phase, 'profile': trial['id'], 'label': trial['label'],
            'settings': {key: trial[key] for key in ('attention', 'kernel', 'compile', 'graph', 'budget', 'mtp')},
            'complete': complete, 'samples_per_cache': samples, 'expected_repeats': 1,
            'observed_kernels': None, 'results': [{'cache': {'cold': stats, 'warm': stats}}]}


class DecisionTrace(unittest.TestCase):
    def config(self):
        return {'search': {'strategy': 'staged', 'objective_cache': 'warm', 'finalists': 2}}

    def evidence(self, records, phases):
        return {'phases': [{'phase': phase, 'samples_per_cache': 20, 'repeats': 1} for phase in phases],
                'records': records, 'baseline': {'id': records[0]['profile']}}

    def test_later_mtp_winner_does_not_rewrite_earlier_budget_decision(self):
        records = [measured('graphs', 1), measured('budgets', 1.01, budget=2048),
                   measured('budgets', 1.02, budget=4096),
                   measured('mtp', .7, budget=8192, mtp=2), measured('mtp', .6, budget=2048, mtp=2)]
        evidence = self.evidence(records, ['graphs', 'budgets', 'mtp'])
        before = copy.deepcopy(evidence)
        steps = build(evidence, self.config())
        budgets, mtp = steps[1:]
        self.assertEqual(budgets['leader']['option'], '8192')
        self.assertEqual([item['option'] for item in budgets['retained']], ['8192', '2048'])
        self.assertEqual(mtp['leader']['settings']['budget'], 2048)
        self.assertEqual(mtp['leader']['option'], 'MTP2')
        old = next(row for row in budgets['groups'][0]['rows'] if row['option'] == '8192')
        self.assertTrue(old['reused'])
        self.assertEqual(old['source_phase'], 'graphs')
        self.assertEqual(old['warm']['e2e_s'], 1)
        self.assertEqual(evidence, before)

    def test_incomplete_or_quality_degraded_faster_result_is_not_leader(self):
        records = [measured('attention', 1), measured('budgets', .1, quality=0, budget=2048),
                   measured('budgets', .01, complete=False, budget=4096)]
        steps = build(self.evidence(records, ['attention', 'budgets']), self.config())
        self.assertEqual(steps[-1]['leader']['option'], '8192')
        self.assertEqual(len(steps[-1]['retained']), 1)
        disqualified = [row for row in steps[-1]['groups'][0]['rows'] if row['option'] != '8192']
        self.assertTrue(all(not row['quality_eligible'] for row in disqualified))

    def test_different_cohort_is_not_reused_for_a_controlled_comparison(self):
        records = [measured('attention', 1, samples=10), measured('budgets', 2, budget=2048)]
        steps = build(self.evidence(records, ['attention', 'budgets']), self.config())
        self.assertEqual([row['option'] for row in steps[-1]['groups'][0]['rows']], ['2048'])

    def test_full_grid_is_not_presented_as_a_staged_decision(self):
        config = self.config()
        config['search']['strategy'] = 'full'
        self.assertEqual(build(self.evidence([measured('full-screen', 1)], ['full-screen']), config), [])

    def test_duplicate_execution_is_explained_without_inventing_measurements(self):
        original = measured('attention', 1)
        duplicate = copy.deepcopy(original)
        trial = make_trial(attention='flashinfer', kernel='auto')
        duplicate.update(phase='kernels', profile=trial['id'], label=trial['label'], complete=False,
                         settings={key: trial[key] for key in original['settings']},
                         results=[{'stage2_status': 'duplicate', 'duplicate_of': original['profile']}])
        evidence = self.evidence([original, duplicate], ['attention', 'kernels'])
        steps = build(evidence, self.config())
        row = next(row for row in steps[-1]['groups'][0]['rows'] if row['profile'] == trial['id'])
        self.assertEqual(row['duplicate_of'], original['profile'])
        self.assertIsNone(row['warm'])
        self.assertFalse(row['quality_eligible'])
        self.assertEqual([item['profile'] for item in steps[-1]['retained']], [original['profile']])
        markdown = stage_markdown(steps[-1])
        self.assertIn('实际执行相同，已合并', markdown)
        self.assertIn('参见 ' + original['label'], markdown)
        evidence.update(decision_steps=steps, objective_cache='warm', complete_screened_candidates=1,
                        screened_candidates=2, skipped=[], recorded_common_settings_by_profile={})
        document = render(evidence, 'GPTQ-W4A16')
        self.assertIn('实际执行相同，已合并；参见 ' + original['label'], document)
        self.assertNotIn('未完成，不参与选择', document)

    def test_render_marks_changed_budget_and_escapes_model_name(self):
        records = [measured('graphs', 1), measured('budgets', 1.01, budget=2048),
                   measured('mtp', .6, budget=2048, mtp=2)]
        evidence = self.evidence(records, ['graphs', 'budgets', 'mtp'])
        evidence.update(decision_steps=build(evidence, self.config()), objective_cache='warm',
                        complete_screened_candidates=3, screened_candidates=3, skipped=[],
                        recorded_common_settings_by_profile={},
                        final_validation={'recommended_trial': make_trial(kernel='humming', attention='flashinfer',
                                                                          budget=2048, mtp=2)})
        document = render(evidence, '<script>not executable</script>')
        self.assertNotIn('<script>', document)
        self.assertIn('&lt;script&gt;', document)
        self.assertIn('本步是 8192 领先，最终配置采用 2048', document)
        self.assertIn('复用前面已测成绩', document)


if __name__ == '__main__':
    unittest.main()
