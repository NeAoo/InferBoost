import json
from pathlib import Path
import tempfile
import unittest

from image.workspace import MANIFEST, fingerprints, initialize


class ImageWorkspaceContracts(unittest.TestCase):
    def test_matching_workspace_preserves_results_and_rejects_modified_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, workspace = root / 'source', root / 'workspace'
            source.mkdir()
            (source / 'quickstart').write_text('original')
            (source / MANIFEST).write_text(json.dumps(fingerprints(source)))
            initialize(source, workspace)
            (workspace / 'runs').mkdir()
            (workspace / 'runs/result.json').write_text('{}')
            initialize(source, workspace)
            self.assertTrue((workspace / 'runs/result.json').is_file())
            (workspace / 'quickstart').write_text('changed')
            with self.assertRaisesRegex(ValueError, '已修改'):
                initialize(source, workspace)

    def test_new_release_does_not_overwrite_old_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, workspace = root / 'source', root / 'workspace'
            source.mkdir()
            (source / 'quickstart').write_text('old')
            (source / MANIFEST).write_text(json.dumps(fingerprints(source)))
            initialize(source, workspace)
            (source / 'quickstart').write_text('new')
            (source / MANIFEST).write_text(json.dumps(fingerprints(source)))
            with self.assertRaisesRegex(ValueError, '其他代码版本'):
                initialize(source, workspace)
            self.assertEqual((workspace / 'quickstart').read_text(), 'old')
