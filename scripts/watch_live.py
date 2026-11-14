"""View an existing local worker with an external UI, preserving frozen source.

For a running older release, copy this script and the new dashboard.py into a
separate tools directory. Run in the worker's container, using its UI Python.
Only reads task state/logs; stop and resume controls are disabled.
"""
import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not (root / 'pipeline/jobs.py').is_file() or not (root / 'jobs' / args.run_id / 'launch.json').is_file():
        parser.error('找不到工作目录或任务；请在原任务容器中运行。')
    # Do not create bytecode in the frozen workspace, even for read-only imports.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    from pipeline.common import identifier
    identifier(args.run_id)
    module_path = Path(__file__).with_name('dashboard.py')
    if not module_path.is_file():
        module_path = Path(__file__).resolve().parents[1] / 'pipeline/dashboard.py'
    spec = importlib.util.spec_from_file_location('pipeline.external_dashboard', module_path)
    viewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viewer)
    viewer.watch(root, args.run_id, read_only=True)


if __name__ == '__main__':
    main()
