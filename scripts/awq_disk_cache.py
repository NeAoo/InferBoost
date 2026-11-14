"""Spill pointwise AWQ captures to disk, preserving the pinned row-selection rule.

Only the current quantization process is patched. Raw captures keep their order,
dtype, shape and bytes. Sequence-sensitive captures use the original path.
This optimizes anonymous RAM, at the cost of temporary disk space and I/O.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import mmap
import os
from pathlib import Path
import tempfile
import threading
import time
import weakref

PROCESSOR_SHA256 = '51be5c05b800fa07cfd6506646575280762762e2273ab0016ae460ad9528b42c'
PATCH_VERSION = 'awq-disk-pointwise-cache-v1'


def row_quotas(lengths, maximum):
    """Exact integer/float operations used by GPTQModel 7.5.0 _pack_token_rows."""
    if maximum <= 0:
        raise ValueError('maximum must be positive')
    total=sum(lengths)
    nonempty=[index for index,length in enumerate(lengths) if length>0]
    quotas=[0]*len(lengths)
    retained=min(total,maximum)
    if retained==total:
        return list(lengths)
    if retained<len(nonempty):
        selected=[0] if retained==1 else [i*(len(nonempty)-1)//(retained-1) for i in range(retained)]
        for index in dict.fromkeys(selected):quotas[nonempty[index]]=1
        return quotas
    effective=[lengths[index] for index in nonempty]
    active=[1]*len(effective)
    remaining=retained-len(effective)
    if remaining>0:
        exact=[length*remaining/total for length in effective]
        floors=[int(share) for share in exact]
        active=[quota+share for quota,share in zip(active,floors)]
        leftover=retained-sum(active)
        order=sorted(range(len(effective)),key=lambda index:(-(exact[index]-floors[index]),index))
        while leftover>0:
            distributed=False
            for index in order:
                if leftover<=0:break
                if active[index]<effective[index]:
                    active[index]+=1;leftover-=1;distributed=True
            if not distributed:break
    for index,quota in zip(nonempty,active):quotas[index]=min(quota,lengths[index])
    return quotas


class DiskCapture:
    def __init__(self,path,shape,dtype,byte_count,owner):
        self.path=Path(path);self.shape=tuple(shape);self.dtype=dtype
        self.byte_count=byte_count
        self._finalizer=weakref.finalize(self,owner.release,self.path,byte_count)

    def dim(self): return len(self.shape)
    def numel(self): return math.prod(self.shape)

    def new_empty(self,shape):
        import torch
        return torch.empty(shape,dtype=self.dtype,device='cpu')

    def selected_rows(self,quota):
        import torch
        hidden=self.shape[-1]
        length=self.numel()//hidden
        if quota<=0:return self.new_empty((0,hidden))
        if self.path.stat().st_size!=self.byte_count:
            raise ValueError('AWQ spill file truncated')
        with self.path.open('rb') as stream:
            # ACCESS_COPY permits a writable buffer without changing the disk file.
            with mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_COPY) as mapping:
                view=torch.frombuffer(mapping,dtype=self.dtype).reshape(length,hidden)
                if quota==length:
                    result=view.clone()
                else:
                    indices=torch.zeros(1,dtype=torch.int64) if quota==1 else torch.arange(quota,dtype=torch.int64)*(length-1)//(quota-1)
                    result=view.index_select(0,indices)
                del view
            if hasattr(os,'posix_fadvise'):
                os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        return result


class DiskStore:
    def __init__(self,directory,max_bytes,rss_limit_bytes):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.max_bytes=max_bytes;self.rss_limit_bytes=rss_limit_bytes
        self.bytes=0;self.peak_bytes=0;self.total_bytes=0;self.captures=0
        self.lock=threading.RLock()

    def put(self,tensor):
        import torch
        tensor=tensor.detach().contiguous()
        if tensor.device.type!='cpu':raise ValueError('Spill requires CPU tensor')
        count=tensor.numel()*tensor.element_size()
        if count==0:return tensor
        with self.lock:
            if self.bytes+count>self.max_bytes:
                raise RuntimeError('AWQ磁盘缓存预算不足，停止量化；不会回退到无限内存缓存')
            status=Path('/proc/self/status')
            if status.exists():
                rss=next(int(line.split()[1])*1024 for line in status.read_text().splitlines() if line.startswith('VmRSS:'))
                if rss>self.rss_limit_bytes:
                    raise MemoryError(f'AWQ进程RSS {rss/1024**3:.1f}GiB超过主动停止阈值')
            descriptor,path=tempfile.mkstemp(prefix='capture-',suffix='.bin',dir=self.directory)
            try:
                with os.fdopen(descriptor,'wb') as stream:
                    stream.write(memoryview(tensor.view(torch.uint8).numpy()).cast('B'))
                    stream.flush()
                    if hasattr(os,'posix_fadvise'):
                        os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            except BaseException:
                Path(path).unlink(missing_ok=True)
                raise
            self.bytes+=count;self.total_bytes+=count;self.captures+=1
            self.peak_bytes=max(self.peak_bytes,self.bytes)
            return DiskCapture(path,tensor.shape,tensor.dtype,count,self)

    def release(self,path,count):
        with self.lock:
            Path(path).unlink(missing_ok=True)
            self.bytes-=count

    def stats(self):
        return {'resident_spill_bytes':self.bytes,'peak_spill_bytes':self.peak_bytes,
                'total_bytes_written':self.total_bytes,'captures_spilled':self.captures}


def pack_disk_rows(tensors,maximum):
    import torch
    hidden=tensors[0].shape[-1]
    if any(tensor.dim()<2 or tensor.shape[-1]!=hidden for tensor in tensors):
        raise ValueError('AWQ缓存形状不一致')
    lengths=[tensor.numel()//hidden for tensor in tensors]
    quotas=row_quotas(lengths,maximum)
    selected=[]
    for tensor,quota,length in zip(tensors,quotas,lengths):
        if quota<=0:continue
        if isinstance(tensor,DiskCapture):
            selected.append(tensor.selected_rows(quota))
        else:
            rows=tensor.reshape(-1,hidden)
            if quota==length:selected.append(rows)
            else:
                indices=torch.zeros(1,dtype=torch.int64) if quota==1 else torch.arange(quota,dtype=torch.int64)*(length-1)//(quota-1)
                selected.append(rows.index_select(0,indices))
    output=torch.cat(selected,dim=0).unsqueeze(0) if selected else tensors[0].new_empty((1,0,hidden))
    return output,sum(lengths)


@contextmanager
def disk_backed_capture(directory,report_path,max_disk_gib=500,rss_limit_gib=180):
    from importlib.metadata import version
    from gptqmodel.looper.awq_processor import AWQProcessor
    source=Path(inspect.getsourcefile(AWQProcessor))
    if version('gptqmodel')!='7.5.0' or hashlib.sha256(source.read_bytes()).hexdigest()!=PROCESSOR_SHA256:
        raise ValueError('AWQ磁盘缓存适配器与当前GPTQModel源码不匹配')
    original_record=AWQProcessor._record_input_feature
    original_pack=AWQProcessor._pack_token_rows
    original_features=AWQProcessor._layer_input_features
    if getattr(original_record,'_exact_cache_dedup',False):
        raise ValueError('磁盘缓存不与旧内存去重补丁叠加')
    store=DiskStore(directory,int(max_disk_gib*1024**3),int(rss_limit_gib*1024**3))
    lock=threading.RLock()

    def record(processor,name,feature,*,dedupe_batch=False):
        # Serialize record+replace so another hook cannot see a half-spilled entry.
        with lock:
            before=len((processor.tasks.get(name) or {}).get('inputs',[]))
            original_record(processor,name,feature,dedupe_batch=dedupe_batch)
            policy=processor._feature_aggregation_policy(name)
            if not policy or policy.get('mode')!='token_rows':return
            entry=processor.tasks.get(name) or {}
            values=entry.get('inputs',[])
            if len(values)>before:
                values[-1]=store.put(values[-1])

    def pack(tensors,*,max_tokens):
        if any(isinstance(tensor,DiskCapture) for tensor in tensors):
            return pack_disk_rows(tensors,max_tokens)
        return original_pack(tensors,max_tokens=max_tokens)

    def features(processor,state):
        before=store.stats();result=original_features(processor,state)
        with Path(report_path).open('a') as stream:
            stream.write(json.dumps({'event':'disk_cache_aggregated','time':time.time(),
                'before':before,'after':store.stats(),'sampling_rule_changed':False})+'\n')
        return result

    AWQProcessor._record_input_feature=record
    AWQProcessor._pack_token_rows=staticmethod(pack)
    AWQProcessor._layer_input_features=features
    try:
        yield store
    finally:
        AWQProcessor._record_input_feature=original_record
        AWQProcessor._pack_token_rows=staticmethod(original_pack)
        AWQProcessor._layer_input_features=original_features
