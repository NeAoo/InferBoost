import json
from pathlib import Path
import tempfile
import unittest

from pipeline.calibration import (OFFICIAL_DATASET, OFFICIAL_REVISION, OFFICIAL_VERSION,
                                  assert_independent, load_calibration, official_defaults)
from pipeline.common import digest
from scripts.prepare_official_calibration import normalize_conversations


class OfficialCalibrationContracts(unittest.TestCase):
    def freeze(self, root, ids=None):
        ids = [7, 8, 9] if ids is None else ids
        cases = [{'id': 'source-1', 'selected_tokens': len(ids),
                  'messages': [{'role': 'user', 'content': 'calibration input'}]}]
        (root / 'cases.jsonl').write_text(json.dumps(cases[0]) + '\n')
        (root / 'tokens.jsonl').write_text(json.dumps({'id': 'source-1', 'input_ids': ids}) + '\n')
        manifest = {'version': OFFICIAL_VERSION, 'source_dataset': OFFICIAL_DATASET,
                    'source_revision': OFFICIAL_REVISION, 'count': 1, 'max_tokens': 4096,
                    'total_tokens': len(ids), 'test_query_overlap': 0, 'tokenizer_sha256': 'tokenizer',
                    'cases_sha256': digest(root / 'cases.jsonl'), 'tokens_sha256': digest(root / 'tokens.jsonl')}
        (root / 'manifest.json').write_text(json.dumps(manifest))

    def test_official_samples_can_be_short_but_cannot_be_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.freeze(root)
            audit, rows = load_calibration(root, 'tokenizer')
            self.assertEqual(rows[0], {'input_ids': [7, 8, 9], 'attention_mask': [1, 1, 1]})
            self.assertEqual(audit['source_revision'], OFFICIAL_REVISION)
            self.freeze(root, [])
            with self.assertRaises(ValueError):
                load_calibration(root, 'tokenizer')

    def test_changed_tokens_or_tokenizer_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.freeze(root)
            with self.assertRaises(ValueError):
                load_calibration(root, 'another-tokenizer')
            (root / 'tokens.jsonl').write_text(json.dumps({'id': 'source-1', 'input_ids': [7, 8, 10]}) + '\n')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                load_calibration(root, 'tokenizer')

    def test_evaluation_prompt_is_excluded_even_with_assistant_turns(self):
        calibration = [{'messages': [{'role': 'user', 'content': 'same  input\ntext'},
                                      {'role': 'assistant', 'content': 'training response'}]}]
        evaluation = [{'messages': [{'role': 'user', 'content': 'same input text'}], 'expected': 'answer'}]
        with self.assertRaisesRegex(ValueError, '相同用户输入'):
            assert_independent(calibration, evaluation)
        evaluation[0]['messages'][0]['content'] = 'independent test question'
        assert_independent(calibration, evaluation)

    def test_official_conversation_roles_and_defaults(self):
        rows = normalize_conversations({'conversations': [{'from': 'human', 'value': 'question'},
                                                           {'from': 'gpt', 'value': 'response'}]})
        self.assertEqual([row['role'] for row in rows], ['user', 'assistant'])
        self.assertEqual(official_defaults()['count'], 512)
        self.assertEqual(official_defaults()['max_tokens'], 4096)
        with self.assertRaises(ValueError):
            normalize_conversations({'conversations': [{'from': 'user', 'value': None}]})


if __name__ == '__main__':
    unittest.main()
