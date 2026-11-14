"""Evidence-qualified recovery hints; an exit code alone does not prove OOM."""
def explain(error):
    text = str(error)
    lowered = text.lower()
    if 'no space left' in lowered or '磁盘' in text:
        return '磁盘空间不足', '保留已有权重，检查工作卷和缓存占用；不要直接清空整个任务目录。'
    if 'cuda out of memory' in lowered:
        return 'GPU显存不足', '确认没有其他GPU进程；降低上下文或服务显存预算，在新任务中测试。'
    if any(term in lowered for term in ['137', 'killed', 'out of memory', 'oom']):
        return '进程被终止，可能内存不足', '先核对cgroup内存事件与平台日志；137不能单独证明OOM。降低编译并发也不能保证量化不超内存。'
    if any(term in lowered for term in ['no such file', 'config.json', '路径', '目录不存在']):
        return '文件或路径不可访问', '检查模型与工作卷是否仍挂载在原路径，确认权重分片和tokenizer完整。'
    if any(term in lowered for term in ['kernel', 'unsupported', '不支持', 'sm70']):
        return '内核或硬件兼容问题', '查看具体候选日志；选择兼容配方或内核，保留其他已完成结果。'
    if any(term in text for term in ['已变化', '代码', '指纹', 'GPU已更换']):
        return '续跑条件发生变化', '恢复原版本、原配置及GPU；环境或数据变化时新建任务，避免混合对照结果。'
    return '任务未完成', '查看阶段日志和退出码。已完成候选保留；从历史任务继续，当前未完成候选可能重做。'
