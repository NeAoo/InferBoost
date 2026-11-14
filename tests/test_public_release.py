"""Public packaging and online/offline calibration contracts; no GPU is used."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from pipeline.workflow import build_configs, parse_arguments, resume_contract
from scripts.package_source import package

ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseContracts(unittest.TestCase):
    def test_online_calibration_has_fixed_revision_and_no_missing_file_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(ROOT / 'configs', root / 'configs')
            shutil.copytree(ROOT / 'data', root / 'data')
            args = parse_arguments(['--root', str(root), '--model', '/model', '--recipes', 'awq4'])
            quant, inference = build_configs(root, args)
            self.assertIsNone(quant['calibration']['raw_data'])
            self.assertEqual(len(quant['calibration']['revision']), 40)
            contract = resume_contract(root, args, quant, inference)
            self.assertEqual(len(contract['data_sha256']), 2)

    def test_explicit_offline_snapshot_is_preserved_and_missing_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(ROOT / 'configs', root / 'configs')
            shutil.copytree(ROOT / 'data', root / 'data')
            path = root / 'configs/quantization.example.json'
            config = json.loads(path.read_text())
            config['calibration']['raw_data'] = 'offline.jsonl'
            path.write_text(json.dumps(config))
            args = parse_arguments(['--root', str(root), '--model', '/model', '--recipes', 'awq4'])
            quant, inference = build_configs(root, args)
            self.assertEqual(quant['calibration']['raw_data'], str((root / 'offline.jsonl').resolve()))
            with self.assertRaises(FileNotFoundError):
                resume_contract(root, args, quant, inference)

    def test_checkout_packaging_excludes_untracked_files_and_rejects_tracked_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'source'
            root.mkdir()
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            (root / 'README.md').write_text('public')
            (root / 'scripts').mkdir()
            (root / 'scripts/.env').write_text('local fixture, never a real credential')
            subprocess.run(['git', '-C', str(root), 'add', 'README.md'], check=True)
            with contextlib.redirect_stdout(io.StringIO()):
                package(root, Path(temporary) / 'package')
            self.assertFalse((Path(temporary) / 'package/scripts/.env').exists())
            outside = Path(temporary) / 'outside.txt'
            outside.write_text('must not be copied')
            (root / 'scripts/link.py').symlink_to(outside)
            subprocess.run(['git', '-C', str(root), 'add', 'scripts/link.py'], check=True)
            with self.assertRaisesRegex(ValueError, '软链接'):
                package(root, Path(temporary) / 'bad-package')


if __name__ == '__main__':
    unittest.main()
