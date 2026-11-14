"""Export a clean GitLab-ready tree without weights, private configs or credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def package(root, output):
    root, output=Path(root).resolve(),Path(output).resolve()
    if output.exists(): raise ValueError('交付目录已存在，请使用新版本目录')
    files=[]
    for directory in ['pipeline','scripts','tests','docs','requirements','image','installer','.github/workflows']:
        files.extend(p for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.extend(root/name for name in [
        'README.md', 'CONTRIBUTING.md', 'CHANGELOG.md', 'LICENSE-NOTICE.md',
        'THIRD_PARTY_NOTICES.md', 'quickstart', 'inferboost', 'install.sh', 'accelerate',
        '.gitignore', '.dockerignore', '.gitlab-ci.yml',
    ])
    if (root / 'LICENSE').is_file():
        files.append(root / 'LICENSE')
    files.extend((root/'configs').glob('*.example.json'))
    files.extend(root/'data'/name for name in ['public100.json','quality100.json'])
    files.extend((root/'references/model-configs').glob('qwen3.*.json'))
    files.extend((root/'references').glob('*.json'))
    # In a checkout, only tracked files can enter the release. In an extracted
    # source distribution the same path/suffix rules still reject credentials.
    tracked = None
    probe = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
    if probe.returncode == 0 and Path(probe.stdout.strip()).resolve() == root:
        tracked = set(subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0'))
    selected = []
    for source in sorted(set(files)):
        relative = str(source.relative_to(root))
        if tracked is not None and relative not in tracked:
            continue
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError('源码包拒绝软链接或越界文件: ' + relative)
        if any(part.startswith(('.env', 'id_rsa', 'id_ed25519')) for part in source.relative_to(root).parts) or source.suffix in ('.pem', '.key', '.p12', '.pfx'):
            raise ValueError('源码包拒绝凭据文件: ' + relative)
        if not source.is_file():
            raise ValueError('交付文件不存在: ' + relative)
        selected.append(source)
    if not selected:
        raise ValueError('没有已跟踪的交付文件，请先把预期源码加入Git索引')
    output.mkdir(parents=True)
    hashes={}
    for source in selected:
        relative=source.relative_to(root)
        destination=output/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        hashes[str(relative)]=hashlib.sha256(destination.read_bytes()).hexdigest()
    (output/'source-manifest.json').write_text(json.dumps({'files':hashes,'weights_included':False,'private_runs_included':False},indent=2)+'\n')
    archive=output.parent/(output.name+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(output,arcname=output.name)
    print(json.dumps({'source_folder':str(output),'archive':str(archive),'files':len(hashes)},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();package(args.root,args.output)
