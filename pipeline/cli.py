import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import sys
import time
from .common import identifier, inside, read_json, write_json, digest, load_config, runtime_environment, source_digest
from . import models, runtime, benchmark, report
from . import deployment


def signal_exit(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')


def input_rows(path):
    rows=read_json(path)
    if not isinstance(rows,list) or not rows:
        raise ValueError('测试数据必须是非空 JSON 数组')
    seen=set()
    for row in rows:
        if not row.get('uid') or row['uid'] in seen or not row.get('group') or not row.get('messages'):
            raise ValueError('每条数据需要唯一uid、group和非空messages')
        seen.add(row['uid'])
    return rows


def signature(root, config):
    dependencies={'dataset':digest(config['benchmark']['dataset'])}
    for key in ['expected_bf16','mtp_reference']:
        if config.get('audit',{}).get(key):
            dependencies[key]=digest(config['audit'][key])
    model_path=config['quantization'].get('model') or config['quantization'].get('source')
    for name in ['config.json','model.safetensors.index.json','quantization_ready.json']:
        path=Path(model_path)/name
        if path.exists(): dependencies['model/'+name]=digest(path)
    identity={'config':config,'files':dependencies,'code':source_digest(root)}
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def run(root, config_path, run_id, resume=False):
    config=load_config(root,config_path)
    run_dir=inside(root,root/'runs'/identifier(run_id))
    fingerprint=signature(root,config)
    if resume:
        existing=read_json(run_dir/'manifest.json')
        if existing['signature']!=fingerprint:
            raise ValueError('代码、配置或输入改变，不能混入旧结果；请使用新的 --run-id')
    else:
        run_dir.mkdir(parents=True,exist_ok=False)
        write_json(run_dir/'config.json',config)
        write_json(run_dir/'manifest.json',{'signature':fingerprint,'started_at':time.time(),
            'code':source_digest(root),'dataset_sha256':digest(config['benchmark']['dataset'])})
        for relative in source_digest(root):
            target=run_dir/'source'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(root/relative,target)
        shutil.copy2(config['benchmark']['dataset'],run_dir/'inputs.json')
    rows=input_rows(run_dir/'inputs.json')
    try:
        with runtime.gpu_lock(root,config):
            write_json(run_dir/'status.json',{'status':'preflight','pid':os.getpid()})
            doctor=runtime.doctor(root,config,run_dir)
            if resume and (run_dir/'gpu_identity.json').exists() and read_json(run_dir/'gpu_identity.json')['uuid']!=doctor['gpu_uuid']:
                raise ValueError('GPU 已更换，不能续跑同一个对比批次')
            write_json(run_dir/'gpu_identity.json',{'uuid':doctor['gpu_uuid']})
            if (run_dir/'model_manifest.json').exists():
                prior=read_json(run_dir/'model_manifest.json')
                actual,_=models.inspect_model(prior['source_model'],payload_hashes=True)
                if actual!=prior['identity']:
                    raise ValueError('模型来源发生变化，拒绝续跑')
                model=Path(prior['load_view'])
                if (run_dir/'execution_config.json').exists():
                    config=read_json(run_dir/'execution_config.json')
            else:
                write_json(run_dir/'status.json',{'status':'preparing_model','pid':os.getpid()})
                model=models.prepare(root,config,run_dir,runtime_environment(root,config))
            for profile in config['profiles']:
                directory=inside(root,run_dir/'profiles'/profile['id'])
                if directory.exists():
                    previous=read_json(directory/'status.json') if (directory/'status.json').exists() else {}
                    if previous.get('status') in ('completed','skipped'):
                        continue
                    directory.rename(directory.with_name(directory.name+'.failed-'+str(time.time_ns())))
                directory.mkdir(parents=True)
                capability=doctor['capabilities'][profile['id']]
                if not capability['eligible']:
                    write_json(directory/'status.json',{'status':'skipped','reason':capability['reason']})
                    continue
                runtime.ensure_idle(config)
                write_json(run_dir/'status.json',{'status':'benchmarking','profile':profile['id'],'pid':os.getpid()})
                write_json(directory/'status.json',{'status':'starting'})
                try:
                    with runtime.server(root,config,profile,model,directory) as (base,_):
                        runtime.warmup(base,rows,config['benchmark'],directory)
                        write_json(directory/'status.json',{'status':'measuring'})
                        benchmark.run_suite(base,rows,config['benchmark'],directory,profile['id'])
                    write_json(directory/'status.json',{'status':'completed','finished_at':time.time()})
                except Exception as error:
                    write_json(directory/'status.json',{'status':'failed','error':str(error)})
                    print(json.dumps({'profile':profile['id'],'error':str(error)},ensure_ascii=False),flush=True)
            result=report.generate(run_dir)
            failed=any(status['status']=='failed' for status in result['statuses'].values())
            write_json(run_dir/'status.json',{'status':'completed_with_failures' if failed else 'completed',
                                             'finished_at':time.time()})
            print(json.dumps({'report':str(run_dir/'report.html'),'recommendation':result['recommendation']},ensure_ascii=False))
            return 1 if failed else 0
    except BaseException as error:
        write_json(run_dir/'status.json',{'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                                        'error':str(error),'time':time.time()})
        raise


def select(root, run_dir, profile_id):
    config=read_json(run_dir/('execution_config.json' if (run_dir/'execution_config.json').exists() else 'config.json'))
    if profile_id=='recommended':
        profile_id=read_json(run_dir/'recommendation.json')['profile']
    if not profile_id:
        raise ValueError('没有满足完整性条件的推荐方案')
    identifier(profile_id)
    matches=[profile for profile in config['profiles'] if profile['id']==profile_id]
    if not matches or read_json(run_dir/'profiles'/profile_id/'status.json')['status']!='completed':
        raise ValueError('只能选择已在本轮完成实测的方案')
    directory=deployment.export(root,run_dir,profile_id)
    write_json(run_dir/'selected.json',{'profile':matches[0],'selected_at':time.time(),
        'quality_validated':False,'validation':'GPU加载与API流完整性、性能；不含业务质量评分'})
    command=[str(root/'quickstart'),'serve','--run-id',run_dir.name]
    (run_dir/'deploy.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.join(command)+' "$@"\n')
    (run_dir/'deploy.sh').chmod(0o755)
    print('部署入口：'+str(run_dir/'deploy.sh'))
    print('完整部署资料：'+str(directory))


def serve(root, run_dir, smoke=False, port=None, host='127.0.0.1', profile_id=None):
    selected_snapshot=None
    if profile_id is None:
        selected_snapshot=read_json(run_dir/'selected.json')['profile']
        profile_id=selected_snapshot['id']
    config,selected,_=deployment.tested_profile(run_dir,profile_id)
    if selected_snapshot is not None and selected_snapshot!=selected:
        raise ValueError('已选参数与实测记录不一致；请新建批次复测')
    if port: config['server']['port']=port
    manifest=read_json(run_dir/'model_manifest.json')
    actual,_=models.inspect_model(manifest['source_model'],payload_hashes=True)
    if actual!=manifest['identity']:
        raise ValueError('选定方案的原模型已变化，拒绝启动')
    directory=inside(root,run_dir/'deployments'/time.strftime('%Y%m%d-%H%M%S'))
    directory.mkdir(parents=True)
    with runtime.gpu_lock(root,config):
        with runtime.server(root,config,selected,manifest['load_view'],directory,host=host,serving=True) as (base,process):
            row={'uid':'deployment-smoke','group':'smoke','messages':[{'role':'user','content':'请用一句中文介绍你能提供什么帮助。'}]}
            result=benchmark.request_one(base,row,{'max_tokens':128,'output_mode':'natural'})
            write_json(directory/'smoke.json',result)
            if result['status']!='success':
                raise RuntimeError('选定方案部署请求未正常结束')
            print(json.dumps({'status':'serving_validated','url':base+'/v1/chat/completions',
                              'answer':result['content'],'smoke_only':smoke},ensure_ascii=False),flush=True)
            if not smoke:
                process.wait()
                raise RuntimeError(f'服务意外退出: {process.returncode}')


def interactive(root):
    stage=input('1=量化与原模型对照，2=推理加速对比 [1]: ').strip() or '1'
    if stage=='1':
        from . import quantization
        return quantization.interactive(root)
    if stage!='2':raise ValueError('请选择1或2')
    default=root/'configs/qwen35-4b.local.json'
    if not default.exists():default=root/'configs/gpu.local.json'
    print('推理加速 Pipeline · 首版支持 AWQ4；其他量化方式后续逐项验证。')
    answer=input(f'配置文件路径 [{default}]: ').strip()
    config_path=Path(answer) if answer else default
    if not config_path.is_file():
        raise ValueError('请先复制 configs/example.json 并配置环境与模型路径')
    config=load_config(root,config_path)
    default_mode='2' if config['quantization']['mode']=='fresh' else '1'
    mode=input(f'1=复用已有AWQ4，2=从完整精度模型重新量化 [{default_mode}]: ').strip() or default_mode
    if mode not in ('1','2'): raise ValueError('请选择1或2')
    config['quantization']['mode']='reuse' if mode=='1' else 'fresh'
    field='model' if mode=='1' else 'source'
    current=config['quantization'].get(field,'')
    config['quantization'][field]=input(f'模型目录 [{current}]: ').strip() or current
    mtp=input('使用MTP草稿进行对比？y/n [n]: ').strip().lower()=='y'
    if mtp:
        current=config.get('mtp_model','')
        config['mtp_model']=input(f'MTP目录 [{current}]: ').strip() or current
    else:
        config['profiles']=[profile for profile in config['profiles'] if not profile.get('mtp')]
    options=[profile['id'] for profile in config['profiles']]
    print('可选方案：'+', '.join(options))
    chosen=input('输入逗号分隔的方案，回车运行全部: ').strip()
    if chosen:
        requested={name.strip() for name in chosen.split(',')}
        if not requested.issubset(options): raise ValueError('方案名称不在当前配置中')
        config['profiles']=[profile for profile in config['profiles'] if profile['id'] in requested]
    run_id='interactive-'+time.strftime('%Y%m%d-%H%M%S')
    path=root/'configs'/(run_id+'.local.json');write_json(path,config)
    result=run(root,path,run_id)
    if result: return result
    selected=input('输入要部署的方案ID，回车只保存报告: ').strip()
    if selected:
        directory=root/'runs'/run_id;select(root,directory,selected);serve(root,directory)
    return 0


def retest(root, run_dir, profile_id, budgets, new_run_id, prepare_only=False):
    import copy
    config, profile, _ = deployment.tested_profile(run_dir,profile_id)
    config=copy.deepcopy(config)
    new_run_id=identifier(new_run_id)
    path=root/'configs'/(new_run_id+'.local.json')
    if path.exists() or (root/'runs'/new_run_id).exists():
        raise ValueError('新批次或配置已存在，请换一个 --new-run-id')
    budgets=list(dict.fromkeys([profile.get('batched_tokens',8192),*budgets]))
    if any(type(value) is not int or value<config['server']['max_num_seqs'] for value in budgets):
        raise ValueError('调度token预算必须是整数，且不小于服务的max_num_seqs')
    if len(budgets)>12:
        raise ValueError('每批最多比较12种预算，请拆分批次')
    manifest=read_json(run_dir/'model_manifest.json')
    config['quantization']['mode']='reuse'
    config['quantization']['model']=manifest['source_model']
    config['profiles']=[]
    for budget in budgets:
        variant=copy.deepcopy(profile)
        variant['id']=profile_id[:70]+'_b'+str(budget)
        variant['batched_tokens']=budget
        variant['label']=report.profile_label(profile).split(' · 预算')[0]+' · 预算'+str(budget)
        config['profiles'].append(variant)
    config['benchmark']['concurrencies']=[1]
    config['benchmark']['cache_modes']=['cold','warm']
    config['report']={'concurrency':1,'cache':'warm'}
    config['provenance']={'parent_run':run_dir.name,'parent_profile':profile_id,
                          'changed_parameter':'batched_tokens','reuses_quantized_weights':True}
    write_json(path,config)
    command=[str(root/'quickstart'),'run','--config',str(path),'--run-id',new_run_id]
    print(json.dumps({'config':str(path),'command':shlex.join(command),'requantization':False},ensure_ascii=False),flush=True)
    if not prepare_only:
        return run(root,path,new_run_id)
    return 0


def main():
    parser=argparse.ArgumentParser(description='量化、加速测试、报告和部署入口')
    parser.add_argument('--root',type=Path,required=True)
    sub=parser.add_subparsers(dest='command')
    p=sub.add_parser('quantize',help='第一阶段：量化并与原模型对照')
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--run-id',default='quantization-'+time.strftime('%Y%m%d-%H%M%S'))
    p.add_argument('--plan',action='store_true',help='只检查模型并展示量化配方，不启动量化')
    p.add_argument('--resume',action='store_true')
    p=sub.add_parser('quantize-report');p.add_argument('--run-id',required=True)
    p=sub.add_parser('quantize-select');p.add_argument('--run-id',required=True);p.add_argument('--candidate',required=True)
    p=sub.add_parser('quantize-catalog');p.add_argument('--model',type=Path,required=True)
    for name in ['run','doctor']:
        p=sub.add_parser(name);p.add_argument('--config',type=Path,required=True)
        p.add_argument('--run-id',default='run-'+time.strftime('%Y%m%d-%H%M%S'))
        if name=='run': p.add_argument('--resume',action='store_true')
    for name in ['status','report','select','serve','export','retest']:
        p=sub.add_parser(name);p.add_argument('--run-id',required=True)
        if name=='report': p.add_argument('--detailed',action='store_true',help='额外生成工程详细报告')
        if name=='select': p.add_argument('--profile',required=True)
        if name=='serve':
            p.add_argument('--smoke-test',action='store_true');p.add_argument('--port',type=int)
            p.add_argument('--host',default='127.0.0.1')
            p.add_argument('--profile',help='启动本轮指定的已测试方案')
        if name=='retest':
            p.add_argument('--profile',required=True)
            p.add_argument('--budgets',nargs='+',type=int,required=True)
            p.add_argument('--new-run-id',required=True)
            p.add_argument('--prepare-only',action='store_true',help='只生成复测配置，不启动GPU任务')
    args=parser.parse_args();root=args.root.resolve()
    signal.signal(signal.SIGTERM,signal_exit)
    if args.command is None: return interactive(root)
    if args.command in ('quantize','quantize-select','quantize-report','quantize-catalog'):
        from . import quantization
        if args.command=='quantize':
            if args.plan:
                print(json.dumps(quantization.plan(root,quantization.load_config(root,args.config)),ensure_ascii=False,indent=2));return 0
            return quantization.run(root,args.config,args.run_id,args.resume)
        if args.command=='quantize-select':
            quantization.select(root,args.run_id,args.candidate);return 0
        if args.command=='quantize-report':
            from .quantization_report import generate
            generate(inside(root,root/'runs'/identifier(args.run_id)));return 0
        from .quantization_catalog import RECIPES,eligibility,inspect_source
        model=inspect_source(args.model)
        print(json.dumps({'model':model,'recipes':{name:eligibility(name,model) for name in RECIPES}},ensure_ascii=False,indent=2));return 0
    if args.command=='run': return run(root,args.config,args.run_id,args.resume)
    directory=inside(root,root/'runs'/identifier(args.run_id))
    if args.command=='doctor':
        directory.mkdir(parents=True,exist_ok=True)
        print(json.dumps(runtime.doctor(root,load_config(root,args.config),directory),ensure_ascii=False,indent=2))
    elif args.command=='status': print(json.dumps(read_json(directory/'status.json'),ensure_ascii=False,indent=2))
    elif args.command=='report': report.generate(directory,detailed=args.detailed)
    elif args.command=='select': select(root,directory,args.profile)
    elif args.command=='serve': serve(root,directory,args.smoke_test,args.port,args.host,args.profile)
    elif args.command=='export':
        profile_id=read_json(directory/'selected.json')['profile']['id']
        print('完整部署资料：'+str(deployment.export(root,directory,profile_id)))
    elif args.command=='retest': return retest(root,directory,args.profile,args.budgets,args.new_run_id,args.prepare_only)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,RuntimeError,OSError,KeyError) as error:
        print('PIPELINE_ERROR: '+str(error),file=sys.stderr)
        raise SystemExit(1)
