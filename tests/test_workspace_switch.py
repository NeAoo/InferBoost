"""Storage changes preserve jobs and data; no GPU or model process is started."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from image import entry, workspace
from pipeline import launcher, storage


class WorkspaceSwitchContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.source = self.base / 'source'
        self.source.mkdir()
        (self.source / 'quickstart').write_text('#!/bin/sh\nexit 0\n')
        (self.source / 'quickstart').chmod(0o755)
        (self.source / workspace.MANIFEST).write_text(json.dumps(workspace.fingerprints(self.source)))
        self.old = self.base / 'old'
        self.new = self.base / 'new workspace'
        self.preference = self.base / 'home' / workspace.PREFERENCE
        workspace.initialize(self.source, self.old)
        (self.old / 'runs').mkdir()
        (self.old / 'runs' / 'result.json').write_text('{"keep": true}')
        workspace.remember(self.source, self.old, self.preference)

    def choose_plain(self, answers):
        with patch('builtins.input', side_effect=answers), patch('sys.stdout', new_callable=io.StringIO):
            return entry.choose_workspace(self.source, self.old, self.preference, current=self.old, plain=True)

    def test_switch_and_switch_back_keep_results_and_background_process(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        try:
            selected = self.choose_plain(['1', '"' + str(self.new) + '"', '1'])
            self.assertEqual(selected, self.new)
            self.assertIsNone(process.poll())
            self.assertEqual((self.old / 'runs/result.json').read_text(), '{"keep": true}')
            self.assertFalse((self.new / 'runs').exists())
            # No default environment override: remembered choice is used.
            self.assertEqual(workspace.resolve_destination(self.source, self.preference, {}), self.new)
            with patch('builtins.input', side_effect=['3', '1']), patch('sys.stdout', new_callable=io.StringIO):
                selected = entry.choose_workspace(self.source, self.new, self.preference, current=self.new, plain=True)
            self.assertEqual(selected, self.old)
            self.assertIsNone(process.poll())
            self.assertTrue((self.old / 'runs/result.json').is_file())
        finally:
            process.terminate()
            process.wait(timeout=10)

    def test_cancel_after_preview_does_not_create_directory_or_change_preference(self):
        before = self.preference.read_bytes()
        self.assertIsNone(self.choose_plain(['1', str(self.new), '3']))
        self.assertFalse(self.new.exists())
        self.assertEqual(self.preference.read_bytes(), before)

    def test_bad_directory_and_permission_failure_leave_preference_unchanged(self):
        before = self.preference.read_bytes()
        model = self.base / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        self.assertIsNone(self.choose_plain(['1', str(model), ':q']))
        self.assertEqual(list(model.iterdir()), [model / 'config.json'])
        with patch('image.workspace.initialize', side_effect=PermissionError('read only')):
            self.assertIsNone(self.choose_plain(['1', str(self.new), '1', ':q']))
        self.assertEqual(self.preference.read_bytes(), before)
        self.assertFalse(self.new.exists())

    def test_invalid_paths_and_overlaps_never_modify_existing_files(self):
        for destination in [self.source, self.source / 'nested', self.old / 'nested', self.base,
                            self.old / 'runs/result.json']:
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                workspace.initialize(self.source, destination, current=self.old)
        self.assertEqual((self.old / 'runs/result.json').read_text(), '{"keep": true}')
        self.assertFalse((self.old / 'nested').exists())

    def test_broken_manifest_and_symlink_escape_are_rejected(self):
        (self.old / workspace.MANIFEST).write_text('broken')
        with self.assertRaisesRegex(ValueError, '其他代码版本'):
            workspace.initialize(self.source, self.old)
        (self.old / workspace.MANIFEST).write_bytes((self.source / workspace.MANIFEST).read_bytes())
        (self.old / 'quickstart').unlink()
        (self.old / 'quickstart').symlink_to(self.source / 'quickstart')
        with self.assertRaisesRegex(ValueError, '已修改'):
            workspace.initialize(self.source, self.old)

    def test_copy_failure_is_atomic_and_only_manifest_files_are_copied(self):
        # Untracked cache in the source tree must not enter a workspace.
        (self.source / 'private-cache').write_text('not part of manifest')
        with patch('image.workspace.shutil.copy2', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                workspace.initialize(self.source, self.new)
        self.assertFalse(self.new.exists())
        self.assertEqual(list(self.base.glob('.inferboost-init-*')), [])
        workspace.initialize(self.source, self.new)
        self.assertFalse((self.new / 'private-cache').exists())

    def test_explicit_environment_wins_and_remembered_release_must_match(self):
        workspace.remember(self.source, self.new, self.preference)
        environment = {'PIPELINE_WORKSPACE': str(self.old)}
        self.assertEqual(workspace.resolve_destination(self.source, self.preference, environment), self.old)
        environment = {'INFERBOOST_DEFAULT_WORKSPACE': '/platform/default', 'PIPELINE_WORKSPACE': '/platform/default'}
        self.assertEqual(workspace.resolve_destination(self.source, self.preference, environment), self.new)
        (self.source / 'quickstart').write_text('new release')
        (self.source / workspace.MANIFEST).write_text(json.dumps(workspace.fingerprints(self.source)))
        self.assertEqual(workspace.resolve_destination(self.source, self.preference, environment), Path('/platform/default'))
        self.assertEqual(workspace.recent_workspaces(self.source, self.preference), [])

    def test_corrupt_preferences_fall_back_and_failed_write_preserves_old_preferences(self):
        before = self.preference.read_bytes()
        with patch.object(Path, 'replace', side_effect=PermissionError('read only')):
            with self.assertRaises(PermissionError):
                workspace.remember(self.source, self.new, self.preference)
        self.assertEqual(self.preference.read_bytes(), before)
        self.preference.write_text('[]')
        self.assertEqual(workspace.resolve_destination(self.source, self.preference, {}), Path(workspace.DEFAULT_WORKSPACE))
        self.preference.write_text('{broken')
        self.assertEqual(workspace.recent_workspaces(self.source, self.preference), [])

    def test_workspace_command_bypasses_invalid_old_workspace_and_cancel_does_not_launch(self):
        with patch.dict(os.environ, {'PIPELINE_WORKSPACE': str(self.old)}), patch('sys.stdin.isatty', return_value=True), \
                patch('image.entry.choose_workspace', return_value=self.new) as choose, patch('image.entry.launch') as launch:
            (self.old / workspace.MANIFEST).write_text('old release')
            entry.enter(self.source, ['workspace', '--plain'])
            choose.assert_called_once()
            launch.assert_called_once_with(self.new, ['--plain'])
        with patch('sys.stdin.isatty', return_value=True), patch('image.entry.choose_workspace', return_value=None), \
                patch('image.entry.launch') as launch:
            self.assertEqual(entry.enter(self.source, ['workspace']), 0)
            launch.assert_not_called()
        with patch('sys.stdin.isatty', return_value=False), self.assertRaisesRegex(ValueError, '交互终端'):
            entry.enter(self.source, ['workspace'])

    def test_plain_menu_switch_is_not_a_workflow_submission(self):
        with patch('sys.stdin.isatty', return_value=True), patch('builtins.input', side_effect=['7', '0']), \
                patch('sys.stdout', new_callable=io.StringIO), patch('pipeline.storage.change_workspace') as change, \
                patch('pipeline.launcher.run_workflow') as submit:
            launcher.main(['--root', str(self.old), '--plain'])
        change.assert_called_once_with(self.old, plain=True)
        submit.assert_not_called()

    def test_source_mode_explains_limit_and_image_switch_execs_new_root(self):
        with patch('image.entry.SOURCE', self.base / 'no-image'), self.assertRaisesRegex(ValueError, '源码'):
            storage.change_workspace(self.source, plain=True)
        with patch('image.entry.SOURCE', self.source), patch('image.entry.choose_workspace', return_value=self.new), \
                patch('image.entry.launch') as launch:
            storage.change_workspace(self.old, plain=True)
        launch.assert_called_once_with(self.new, ['--plain'])

    def test_rich_choice_and_cancel_keep_same_contract(self):
        try:
            from pipeline import ui
        except ImportError:
            self.skipTest('optional terminal dependencies not installed')
        with patch.object(ui, 'choose', side_effect=['path', 'use']), patch.object(ui, 'text_input', return_value=str(self.new)), \
                patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(entry.choose_workspace(self.source, self.old, self.preference, current=self.old), self.new)
        before = self.preference.read_bytes()
        with patch.object(ui, 'choose', side_effect=ui.Cancel), patch('sys.stdout', new_callable=io.StringIO):
            self.assertIsNone(entry.choose_workspace(self.source, self.new, self.preference, current=self.new))
        self.assertEqual(self.preference.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
