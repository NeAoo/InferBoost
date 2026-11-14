# InferBoost：沙盒验收

## 开始前：默认开启的加速

- 普通KV Cache：生成时复用历史Key/Value，减少重复计算。
- 前缀缓存：复用相同输入前缀；没有重复前缀时不一定受益。
- 分块预填充：长输入分块处理，便于调度。
- KV精度auto：由框架按模型决定，与W/A量化精度不同。

默认搜索不对这些项分别做开关消融。MTP草稿保持原精度、不参与主模型量化；没有输入MTP目录时不进行MTP搜索。
后续测试的是注意力、量化矩阵内核、编译和CUDA Graph、调度预算、MTP的兼容组合；用户看结果后选择。

## 导入和启动

候选镜像为 `inferboost:0.5.0-cu130`，Linux amd64，内含量化与推理环境以及公共测试数据，不含模型权重。
依赖沿用已验收运行时：vLLM0.33.0 / Torch2.14.0 / CUDA13.0；更换GPU、驱动或模型仍需真实验收。

在有Docker与NVIDIA Container Toolkit的GPU沙盒中：

```bash
docker run -d --name inferboost-work --gpus all --shm-size=16g \
  -v /absolute/model-directory:/models:ro \
  -v /absolute/persistent-directory:/workspace \
  inferboost:0.5.0-cu130 idle

# 打开菜单；退出查看后容器和后台任务保留。
docker exec -it inferboost-work quickstart
```

云平台直接导入镜像时，在平台上挂载模型和可写持久卷，然后进入终端：

```bash
# 将路径换成平台实际挂载目录；首次使用应是新的空目录。
export PIPELINE_WORKSPACE=/mnt/persistent/inferboost-v050
quickstart
```

需要更换存储位置时选择主菜单“工作目录 · 更换存储路径”，或运行 `quickstart workspace`。旧目录的任务与结果保持原位置；新镜像不能直接覆盖旧版本工作区。[存储切换规则](terminal-interaction.md#更换存储路径)

量化与搜索适合开发实例或批处理任务，测试服务会随候选启停；部署阶段提供持续推理服务。按所选运行方式配置容器入口和健康检查，见[资源与端口配置](configuration.md#运行方式与端口)。平台定制入口见[领慧适配](platforms/linghui.md)。无 GPU 时可浏览菜单与报告，量化和推理需要兼容 GPU。

## 终端菜单

首次启动会选择工作目录并自动显示GPU、内存、磁盘；进入“新模型”后填写模型和可选MTP。支持路径Tab补全、方向键/空格多选、返回修改、确认页与后台看板。[交互说明](terminal-interaction.md)

先用“快速体验”验证原模型与一个兼容量化配方的完整链路；通过后新建“标准对比”任务，使用完整数据和三轮性能。已有量化模型可直接进入第二阶段。配方兼容检查不能替代实际加载，原模型也必须能放入单卡。

量化完成、加速完成分别等待用户选择；看板按C选择，按R看结果。只有明确授权“按推荐自动继续”才自动选择。部署包会做一次真实请求并退出，持续服务从“启动已有部署包”启动。

**Q退出查看不停止任务；S确认才停止。** 关闭终端后重新运行`quickstart`，从任务中心重进看板。容器重建后挂载原卷、使用相同版本续跑；未完成量化可能重做。资源预算、编译并发与上下文设置见[配置指南](configuration.md)。

## 每个产物放在哪里

```text
$PIPELINE_WORKSPACE/
├── runs/<任务名>-quant/
│   ├── candidates/awq4/quantized/     新量化后的真实完整权重
│   ├── candidates/gptq4/quantized/    另一种配方的真实完整权重
│   ├── candidates/<方案>/model/      加载视图，权重通常是软链接
│   ├── candidates/<方案>/model_manifest.json
│   ├── selected_model.json          用户选择及第二步交接清单
│   └── report.md / report.html       量化质量、资源、速度
├── runs/inference/<任务名>-accel/    初筛、完整复测、接受率及报告
├── runs/workflows/<任务名>/
│   ├── request.json                 参数与来源，供菜单续跑
│   └── deployment/                  start.sh与部署配置（不含权重）
└── cache/                           编译与运行缓存
```

`original`参照不复制原模型；复用模式也可能引用旧批次权重。**真实位置以 `model_manifest.json` 的 `artifact` 字段为准。**
迁移请复制真实权重目录和部署包，不要只复制带绝对软链接的 `model/`。
跨实例保留需要重挂同一持久卷，`docker commit`或只保存容器镜像不会带走挂载卷里的结果。
升级组件使用新工作目录，保留旧实验及其代码；相同版本可重复进入，原有产物不会被清空。

## 验收记录

1. 首次设置和环境自检能看到实际GPU、内存、持久卷及依赖。
2. “新模型”完成原模型、至少一种量化的独立评分，真实权重出现于`quantized/`。
3. 选择模型后完成加速搜索；HTML/Markdown顶部有默认加速说明，各阶段有跳过原因。
4. MTP开启时接受率注明统计范围；缺指标时明确显示未采集。
5. 选择配置后生成部署包，并看到真实请求成功记录。
6. 从“启动已有部署包”启动持续服务或请求验收，检查响应及`/health`。
7. 运行中Q退出并重新进入任务中心，计算继续；停止确认后服务清理，C可续跑。
8. 使用同一持久目录重新进入，仍可看到任务、权重和报告；标准性能3轮无缺失，质量评分没有重复计为更多题。

遇到问题保留任务名、`runs/workflows/<任务>/status.json`、失败阶段日志及环境自检输出即可定位。
环境自检、菜单和 CPU 测试用于验证入口与编排；量化、加载和推理结果以目标 GPU 上的任务报告为准。
