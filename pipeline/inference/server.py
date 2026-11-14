"""Launch one frozen candidate under the official sweep's process ownership."""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import time

from pipeline.common import read_json, write_json
from .config import environment


def assert_idle(config, wait_seconds=40):
    deadline = time.monotonic() + wait_seconds
    while True:
        gpu = subprocess.check_output(['nvidia-smi', '-i', str(config['runtime'].get('gpu', 0)),
                                       '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        with socket.socket() as connection:
            busy_port = connection.connect_ex(('127.0.0.1', config['server']['port'])) == 0
        if not gpu and not busy_port:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('GPU或端口仍被占用，不会停止其他服务: ' + gpu)
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--profile-id', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    run = args.run_dir
    phase = read_json(run / 'phases' / (args.phase + '.json'))
    candidates = {candidate['id']: candidate for candidate in phase['candidates']}
    if args.profile_id not in candidates:
        raise ValueError('候选不在本轮冻结计划中')
    candidate = candidates[args.profile_id]
    config = read_json(run / 'config.json')
    if args.host != '127.0.0.1' or args.port != config['server']['port']:
        raise ValueError('官方sweep探测地址与冻结服务配置不一致')
    model = read_json(run / 'model.json')
    directory = run / 'trials' / args.phase / args.profile_id
    directory.mkdir(parents=True, exist_ok=True)
    assert_idle(config)
    env = environment(Path(config['root']), config, run / 'source')
    env['STAGE2_REFERENCES'] = __import__('json').dumps(model['references'])
    env['STAGE2_MTP'] = str(candidate['mtp'])
    env['STAGE2_AUDIT_FILE'] = str(directory / 'audit.json')
    env['VLLM_SERVER_DEV_MODE'] = '1'
    write_json(directory / 'launch.json', {'profile': candidate, 'command': candidate['command'],
                                         'adapter': 'vllm-0.33-stage2-v1'})
    write_json(directory / 'process.json', {'pid': os.getpid(),
                                          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
    write_json(run / 'active.json', {'phase': args.phase, 'profile_id': args.profile_id})
    log = os.open(directory / 'server.log', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(log, 1)
    os.dup2(log, 2)
    os.close(log)
    os.execvpe(candidate['command'][0], candidate['command'], env)


if __name__ == '__main__':
    main()
