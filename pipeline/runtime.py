"""GPU ownership, capability gating and managed vLLM lifecycle."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from .common import capture, read_json, write_json, append_json, runtime_environment
from .benchmark import endpoint, request_one
from .resources import gpu_lease
from .terminal import Progress


def doctor(root, config, directory):
    environment = runtime_environment(root, config)
    code = '''import json,torch,importlib.metadata as m
torch.ones(1,device="cuda").sum().item()
print(json.dumps({"gpu":torch.cuda.get_device_name(),"capability":torch.cuda.get_device_capability(),"torch":torch.__version__,"cuda":torch.version.cuda,"vllm":m.version("vllm"),"flashinfer":m.version("flashinfer-python")}))'''
    result = capture([config['runtime']['python'], '-c', code], env=environment, timeout=120)
    info = json.loads(result.splitlines()[-1])
    info['gpu_uuid'] = capture(['nvidia-smi','-i', str(config['runtime'].get('gpu',0)),
                               '--query-gpu=uuid','--format=csv,noheader'])
    info['capabilities'] = {}
    for profile in config['profiles']:
        reason = None
        if profile['attention']=='fa3' and info['capability'][0] != 9:
            reason = 'FA3 需要 Hopper；本轮 GPU 为 SM' + ''.join(map(str,info['capability']))
        info['capabilities'][profile['id']] = {'eligible':reason is None, 'reason':reason}
    if info['vllm'] != '0.33.0':
        raise ValueError('本版启动适配器已锁定 vLLM 0.33.0，请使用匹配环境')
    write_json(directory/'doctor.json',info)
    return info


@contextmanager
def gpu_lock(root, config):
    with gpu_lease(root, config['runtime'].get('gpu', 0)):
        ensure_idle(config)
        yield


def ensure_idle(config):
    output = capture(['nvidia-smi','-i', str(config['runtime'].get('gpu',0)),
                      '--query-compute-apps=pid','--format=csv,noheader'])
    if output.strip():
        raise RuntimeError('GPU 已有计算进程；不会停止现有服务: '+output)


def build_command(config, profile, model, port, host='127.0.0.1'):
    settings = config['server']
    attention = {'backend':'FLASHINFER'} if profile['attention']=='flashinfer' else {
        'backend':'FLASH_ATTN', 'flash_attn_version':3 if profile['attention']=='fa3' else 2}
    command = [config['runtime']['python'],'-m','vllm.entrypoints.openai.api_server',
        '--model',str(model),'--host',host,'--port',str(port),'--served-model-name','pipeline-model',
        '--dtype',settings.get('dtype','bfloat16'),'--tensor-parallel-size','1',
        *(['--quantization',settings.get('quantization','awq_marlin')]
          if settings.get('quantization','awq_marlin') is not None else []),
        '--max-model-len',str(settings['max_model_len']),
        '--max-num-seqs',str(settings['max_num_seqs']),
        '--max-num-batched-tokens',str(profile.get('batched_tokens',8192)),
        '--gpu-memory-utilization',str(settings.get('gpu_memory_utilization',0.90)),
        '--enable-chunked-prefill','--language-model-only','--reasoning-parser','qwen3',
        '--attention-config',json.dumps(attention),
        '--override-generation-config',json.dumps({'max_new_tokens':settings.get('max_output_tokens',16384)})]
    if config.get('tokenizer'):
        command += ['--tokenizer',config['tokenizer']]
    if profile.get('prefix_cache',True):
        command += ['--enable-prefix-caching']
    else:
        command += ['--no-enable-prefix-caching']
    if profile.get('graph',True):
        command += ['--compilation-config',json.dumps({'mode':3,'cudagraph_mode':'FULL_AND_PIECEWISE'})]
    else:
        command += ['--enforce-eager']
    if profile.get('mtp'):
        command += ['--speculative-config',json.dumps({'method':'mtp','model':config['mtp_model'],
                                                      'num_speculative_tokens':profile['mtp']})]
    if config.get('audit',{}).get('expected_bf16'):
        command += ['--worker-cls',settings.get('worker_cls','pipeline.audit_worker.AuditedWorker')]
    return command


def monitor(directory, gpu, stop):
    while not stop.is_set():
        result = subprocess.run(['nvidia-smi','-i',str(gpu),
            '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem',
            '--format=csv,noheader,nounits'],capture_output=True,text=True)
        append_json(directory/'gpu.jsonl',{'timestamp':time.time(),'sample':result.stdout.strip(),
                                         'returncode':result.returncode})
        stop.wait(5)


@contextmanager
def server(root, config, profile, model, directory, host='127.0.0.1', serving=False, cancelled=None):
    port = config['server']['port']
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port))==0:
            raise RuntimeError(f'端口 {port} 已占用')
    environment = runtime_environment(root,config)
    if serving:
        environment['VLLM_SERVER_DEV_MODE']='0'
    audit = config.get('audit',{})
    if audit.get('expected_bf16'):
        environment.update(EXPECTED_BF16_FILE=audit['expected_bf16'],
                           LOADING_AUDIT_FILE=str(directory/'loading_audit.json'))
        if profile.get('mtp') and audit.get('mtp_reference'):
            environment['MTP_AUDIT_REFERENCE']=audit['mtp_reference']
        else:
            environment.pop('MTP_AUDIT_REFERENCE',None)
    command = build_command(config,profile,model,port,host)
    write_json(directory/'launch.json',{'command':command, 'profile':profile,
        'environment':{key:environment[key] for key in ['CUDA_VISIBLE_DEVICES','CUDA_HOME',
            'LD_LIBRARY_PATH','MAX_JOBS','VLLM_CACHE_ROOT','FLASHINFER_WORKSPACE_BASE'] if key in environment}})
    log = (directory/'server.log').open('w')
    process = subprocess.Popen(command,env=environment,cwd=root,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True)
    write_json(directory/'process.json',{'pid':process.pid,'coordinator':os.getpid(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'started_at':time.time()})
    stop = threading.Event()
    telemetry = threading.Thread(target=monitor,args=(directory,config['runtime'].get('gpu',0),stop),daemon=True)
    telemetry.start()
    def cancellation_watch():
        while not stop.wait(.5):
            if cancelled is not None and cancelled.is_set():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                return
    watcher = threading.Thread(target=cancellation_watch, daemon=True)
    watcher.start()
    base = f'http://127.0.0.1:{port}'
    started = time.monotonic()
    try:
        with Progress('启动 vLLM · ' + profile['id']) as startup:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'vLLM 启动退出 {process.returncode}，查看 {directory}/server.log')
                try:
                    endpoint(base,'/health',timeout=2)
                    break
                except (OSError, TimeoutError):
                    if time.monotonic()-started>config['server'].get('startup_timeout_seconds',1800):
                        raise TimeoutError('vLLM 启动超时')
                    startup.update(detail='加载/编译/预热；详见 server.log')
                    time.sleep(2)
        if audit.get('expected_bf16'):
            result = read_json(directory/'loading_audit.json')
            if not result.get('passed') or result['versions']['vllm']!='0.33.0':
                raise RuntimeError('真实模型加载审计失败')
        write_json(directory/'ready.json',{'startup_seconds':time.monotonic()-started,
                                          'models':json.loads(endpoint(base,'/v1/models'))})
        yield base, process
    finally:
        try:
            os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait(timeout=15)
        stop.set()
        telemetry.join(timeout=10)
        watcher.join(timeout=2)
        log.close()
        write_json(directory/'cleanup.json',{'exit_code':process.returncode,'finished_at':time.time()})
        # Workers may need a few seconds to release CUDA memory.
        for _ in range(30):
            try:
                ensure_idle(config)
                break
            except RuntimeError:
                time.sleep(1)


def warmup(base, rows, settings, directory):
    from concurrent.futures import ThreadPoolExecutor
    from .benchmark import reset_cache
    for group in dict.fromkeys(row['group'] for row in rows):
        candidates = [row for row in rows if row['group']==group]
        row = max(candidates,key=lambda item:item.get('input_tokens') or len(json.dumps(item['messages'])))
        reset_cache(base)
        result=request_one(base,row,settings,warmup=True)
        append_json(directory/'warmup.jsonl',result)
        if result['status']!='fixed_complete':
            raise RuntimeError('该任务类型预热失败')
    # Exercise each tested scheduler concurrency before starting any timed cohort.
    ordered=sorted(rows,key=lambda row:row.get('input_tokens') or len(json.dumps(row['messages'])),reverse=True)
    for concurrency in settings['concurrencies']:
        reset_cache(base)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results=list(executor.map(lambda row:request_one(base,row,settings,warmup=True),ordered[:concurrency]))
        for result in results:
            append_json(directory/'warmup.jsonl',dict(result,concurrency=concurrency))
        if any(result['status']!='fixed_complete' for result in results):
            raise RuntimeError('并发预热失败')
