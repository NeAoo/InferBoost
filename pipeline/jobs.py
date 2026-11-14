"""Session-independent local workers; UI detaches, explicit stop requests clean up."""
import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

from .common import identifier, read_json, source_digest, write_json

CHILDREN = {}


def reap_children():
    for pid, process in list(CHILDREN.items()):
        if process.poll() is not None:
            CHILDREN.pop(pid, None)


def read_optional(path):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return {}


def process_identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat')
        if stat.is_file():
            fields = stat.read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                return None
            return Path('/proc/sys/kernel/random/boot_id').read_text().strip() + ':' + fields[19]
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=', '-o', 'lstart='],
                                capture_output=True, text=True, timeout=2)
        value = result.stdout.strip()
        return value.split(maxsplit=1)[1] if result.returncode == 0 and value and not value.startswith('Z') else None
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def alive(record):
    return bool(record.get('pid') and record.get('identity')
                and process_identity(record['pid']) == record['identity'])


@contextmanager
def job_lock(directory, blocking=False):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'worker.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError('这个任务仍在运行，请进入看板查看。') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def argument_value(arguments, key, default=None):
    return arguments[arguments.index(key) + 1] if key in arguments else default


def submit(root, arguments):
    reap_children()
    root = Path(root).resolve()
    run_id = identifier(argument_value(arguments, '--run-id'))
    directory = root / 'jobs' / run_id
    with job_lock(directory):
        launch = read_optional(directory / 'launch.json')
        if alive(launch):
            raise RuntimeError('这个任务仍在运行，请进入看板查看。')
        workflow = root / 'runs/workflows' / run_id
        if workflow.exists() and '--resume' not in arguments:
            raise ValueError('任务已存在；请从历史任务继续，或更换名称。')
        if '--resume' in arguments and not (workflow / 'request.json').is_file():
            raise ValueError('任务尚未创建完整流程记录，请以原参数重新提交。')
        attempt = uuid.uuid4().hex
        request = {'arguments': list(arguments), 'run_id': run_id, 'attempt': attempt,
                   'created_at': time.time(), 'source': source_digest(root)}
        write_json(directory / 'request.json', request)
        log_path = directory / (attempt + '.log')
        environment = dict(os.environ, PYTHONPATH=str(root), PYTHONUNBUFFERED='1', NO_COLOR='1')
        with log_path.open('a') as log:
            process = subprocess.Popen([sys.executable, '-m', 'pipeline.jobs', 'worker',
                '--root', str(root), '--run-id', run_id], cwd=root, env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        CHILDREN[process.pid] = process
        record = {'pid': process.pid, 'identity': process_identity(process.pid),
                  'attempt': attempt, 'log': str(log_path), 'submitted_at': time.time()}
        write_json(directory / 'launch.json', record)
    return run_id


def status(root, run_id):
    reap_children()
    directory = Path(root) / 'jobs' / identifier(run_id)
    launch = read_optional(directory / 'launch.json')
    state = read_optional(directory / 'state.json')
    if state.get('attempt') != launch.get('attempt'):
        state = {'state': 'starting', 'attempt': launch.get('attempt')}
    running = alive(launch)
    if not running and state.get('state') in ('running', 'starting', 'stopping'):
        state = dict(state, state='interrupted', error='任务进程已离开；实例重启或进程被终止，检查日志后续跑。')
    return {**launch, **state, 'alive': running,
            'workflow': read_optional(Path(root) / 'runs/workflows' / run_id / 'status.json')}


def request_stop(root, run_id):
    current = status(root, run_id)
    if not current['alive']:
        raise ValueError('任务已经停止，无需再次发送停止请求。')
    write_json(Path(root) / 'jobs' / identifier(run_id) / 'stop.json',
               {'attempt': current['attempt'], 'requested_at': time.time()})


def continuation(root, run_id, candidate=None, profile=None):
    from .launcher import resume_arguments
    arguments = resume_arguments(Path(root), run_id)
    # Keep selections made during a previous detached attempt.
    previous = read_optional(Path(root) / 'jobs' / identifier(run_id) / 'request.json').get('arguments', [])
    for key, value in [('--candidate', candidate or argument_value(previous, '--candidate')),
                       ('--profile', profile or argument_value(previous, '--profile'))]:
        if value:
            if key in arguments:
                arguments[arguments.index(key) + 1] = value
            else:
                arguments += [key, value]
    return arguments


def worker(root, run_id):
    directory = root / 'jobs' / identifier(run_id)
    with job_lock(directory, blocking=True):
        request = read_json(directory / 'request.json')
        if request['source'] != source_digest(root):
            write_json(directory / 'state.json', {'state': 'failed', 'attempt': request['attempt'],
                       'error': '提交后源码发生变化；使用原代码续跑或新版本工作区。'})
            return 1
        stopped = threading.Event()
        state = {'state': 'running', 'attempt': request['attempt'], 'started_at': time.time()}
        def heartbeat():
            stop_sent = False
            while not stopped.is_set():
                stop_request = read_optional(directory / 'stop.json')
                if stop_request.get('attempt') == request['attempt'] and not stop_sent:
                    state['state'] = 'stopping'
                    stop_sent = True
                    os.kill(os.getpid(), signal.SIGTERM)
                write_json(directory / 'state.json', dict(state, heartbeat_at=time.time()))
                stopped.wait(1)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        thread = threading.Thread(target=heartbeat, daemon=True)
        code = 0
        try:
            thread.start()
            from .workflow import main
            main(['--root', str(root), *request['arguments']], install_signals=False)
            workflow = read_optional(root / 'runs/workflows' / run_id / 'status.json')
            state['state'] = workflow.get('status', 'failed')
        except KeyboardInterrupt:
            code = 130
            state.update(state='interrupted', error='任务已停止；已完成候选保留，当前候选可能需要重做。')
        except BaseException as error:
            code = 1
            state.update(state='failed', error=str(error))
            import traceback
            traceback.print_exc()
        finally:
            stopped.set()
            if thread.ident is not None:
                thread.join(timeout=3)
            write_json(directory / 'state.json', dict(state, finished_at=time.time(), exit_code=code))
        return code


def main():
    parser = argparse.ArgumentParser(description='InferBoost 后台任务与断线续看')
    parser.add_argument('command', choices=['worker', 'status', 'watch', 'stop'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if args.command == 'worker':
        return worker(args.root, args.run_id)
    if args.command == 'stop':
        request_stop(args.root, args.run_id)
    elif args.command == 'watch':
        from .dashboard import watch
        watch(args.root, args.run_id)
    else:
        import json
        print(json.dumps(status(args.root, args.run_id), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
