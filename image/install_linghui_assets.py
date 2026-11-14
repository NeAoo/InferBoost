"""Verify platform-provided assets before installing into the derived image."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

root = Path(sys.argv[1])
record = json.loads(Path('/opt/pipeline/references/linghui-assets.json').read_text())
for asset in record['assets']:
    path = root / asset['file']
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != asset['sha256']:
        raise RuntimeError('平台构建资产校验失败：' + asset['file'])
subprocess.run(['dpkg', '-i', str(root / 'code-server_4.14.1_amd64.deb')], check=True)
unpacked = Path(tempfile.mkdtemp(prefix='inferboost-extensions-'))
with tarfile.open(root / 'vscode-ext.tar.gz') as archive:
    for member in archive.getmembers():
        destination = (unpacked / member.name).resolve()
        if not destination.is_relative_to(unpacked) or not (member.isfile() or member.isdir()):
            raise ValueError('扩展归档包含不允许的路径或文件类型')
    archive.extractall(unpacked)
packages = sorted(unpacked.rglob('*.vsix'))
if not packages:
    raise RuntimeError('官方扩展包中未找到vsix')
for path in packages:
    subprocess.run(['/usr/bin/code-server', '--extensions-dir', '/opt/inferboost-vscode-extensions',
                    '--install-extension', str(path), '--force'], check=True)
shutil.rmtree(unpacked)
