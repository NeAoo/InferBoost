"""Verify an exported source tree, archive bytes and relative documentation links.

Uses only the Python standard library. This checks distribution integrity, not
GPU behavior, licensing, external URLs or Markdown anchor rendering.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
from urllib.parse import unquote, urlsplit


MANIFEST_NAME = 'source-manifest.json'
REQUIRED_FILES = (
    'README.md', 'CONTRIBUTING.md', 'CHANGELOG.md', 'LICENSE-NOTICE.md',
    'THIRD_PARTY_NOTICES.md', '.gitlab-ci.yml', 'quickstart',
    'docs/assets/workflow-overview.png', 'docs/sandbox-quickstart.md',
    'docs/release-readiness.md', 'image/Dockerfile.update',
)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, archive=None):
    root = Path(root).resolve()
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    expected = manifest['files']
    if not isinstance(expected, dict) or not expected:
        raise ValueError('Source manifest must contain a nonempty file mapping')
    actual = {str(path.relative_to(root)): path for path in root.rglob('*')
              if path.is_file() and '__pycache__' not in path.parts
              and path != root / MANIFEST_NAME}
    errors = []
    for name in sorted(set(expected) - set(actual)):
        errors.append('missing: ' + name)
    for name in sorted(set(actual) - set(expected)):
        errors.append('unlisted: ' + name)
    for name in sorted(set(expected) & set(actual)):
        if actual[name].is_symlink():
            errors.append('unexpected symlink: ' + name)
        elif fingerprint(actual[name]) != expected[name]:
            errors.append('checksum mismatch: ' + name)
    for name in REQUIRED_FILES:
        if name not in actual:
            errors.append('missing required release file: ' + name)

    checked_links = 0
    for name, document in actual.items():
        if document.suffix != '.md':
            continue
        for target in re.findall(r'!?\[[^\]\n]*\]\(([^)\n]+)\)', document.read_text()):
            parsed = urlsplit(target.strip().strip('<>'))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            destination = (document.parent / unquote(parsed.path)).resolve()
            checked_links += 1
            if not destination.is_relative_to(root) or not destination.exists():
                errors.append('broken or external local link: ' + name + ' -> ' + target)

    if archive is not None:
        archived = {}
        with tarfile.open(archive, 'r:gz') as handle:
            for member in handle.getmembers():
                if member.isdir():
                    continue
                path = Path(member.name)
                if not member.isfile() or not path.parts or path.parts[0] != root.name:
                    errors.append('unexpected archive member: ' + member.name)
                    continue
                relative = str(Path(*path.parts[1:]))
                if relative in archived:
                    errors.append('duplicate archive member: ' + relative)
                with handle.extractfile(member) as stream:
                    archived[relative] = hashlib.sha256(stream.read()).hexdigest()
        archive_expected = {**expected, MANIFEST_NAME: fingerprint(root / MANIFEST_NAME)}
        if archived != archive_expected:
            errors.append('archive bytes do not match the source manifest and tree')

    if errors:
        raise ValueError('\n'.join(errors))
    return {'status': 'passed', 'source_files': len(expected),
            'relative_document_links': checked_links, 'archive_checked': archive is not None,
            'gpu_validation': 'not_performed', 'license_audit': 'not_performed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.root, args.archive), ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, tarfile.TarError) as error:
        parser.exit(1, 'PACKAGE_ERROR: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
