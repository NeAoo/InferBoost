# 参与 InferBoost 开发

修改通过合并请求评审，公开许可状态见 [LICENSE-NOTICE.md](LICENSE-NOTICE.md)。

## 本地开发

使用 Python 3.11。CPU 契约测试不要求 CUDA，量化和真实推理需要提供的 GPU 运行时镜像。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes --no-deps -r requirements/terminal.lock
./quickstart --version
python -m unittest discover -s tests -q
```

源码运行时复制 `configs/*.example.json` 为自己的 `*.local.json`，配置量化 / 推理 Python 路径；详细入口见 [CLI 手册](docs/cli-reference.md)。不要直接修改已完成实验的 `runs/.../source/` 快照。

## 修改应该放在哪里

- 模型适配、敏感层和量化准入：`pipeline/quantization_catalog.py`。
- 量化后端调用：`scripts/quantize_compressor.py`，保持 MTP 原精度校验。
- 硬件兼容和加速组合：`pipeline/inference/planning.py`。
- 执行、复测及服务清理：`pipeline/inference/execution.py`、`server.py`、`worker.py`。
- 展示：使用已保存的结构化结果，修改 HTML / Markdown 渲染，不改变原始测量值。
- 数据、镜像和公共入口变动：同步示例、README、打包清单和相应验证。

完整模块关系见 [架构地图](docs/architecture.md)。优先复用现有边界，不把量化、搜索、服务启动集中到一个大脚本。

## 提交合并请求

说明具体问题、改变后的行为、复现命令和验证结果。运行相关 CPU 测试；行为涉及模型加载、精度或 GPU 内核时，还要提供相应 GPU 实测记录。仅变更文档时检查链接和命令；不要为了文案增加无意义单元测试。

新增适配需记录模型 `model_type`、硬件、驱动、运行时、量化方法和 W/A 精度。分别标明：已实现、通过准入、导出成功、加载成功、质量通过。设备不支持应给出跳过原因，不静默回退到另一精度。

性能对比需保留数据身份、生成参数、并发、预热、cold/warm、重复次数、完整请求数和失败项。只比较相同条件，展示收益也展示退化；不要用一组硬件结果推广到所有硬件。

## 可提交与不应提交的内容

提交代码、公开示例、合成测试数据和文档。私有配置、用户权重、校准 token 缓存、业务请求、日志、凭据及历史运行目录不进入公共仓库。不要提交完整开发目录，先使用 [干净源码打包](docs/image-delivery.md)。

默认 GitLab CI 运行 CPU 测试、包完整性及本地文档链接检查。它在代码推送后触发，镜像构建和目标 GPU 验收另外执行，见[发布流程](docs/release-readiness.md)。

## 报告问题

在项目建立的 issue / 合并请求中附上：版本与镜像 ID、GPU 与驱动、失败阶段、脱敏命令、退出码、最小复现及相关错误日志。不要上传 token、内网认证信息或原始业务数据。通过仓库协作入口提交。

## 公共文档约定

- README 说明输入、操作与输出；参数、资源和支持边界放在专题文档。
- 示例使用可替换的通用路径，平台定制说明放在 `docs/platforms/`。
- 区分默认值、可配置项和硬件硬约束，内容以代码与配置为准。
- 特定模型的分数、性能、运行编号和故障排查过程保存在任务报告或交付记录，不作为通用用户指南。
- 行为变动时同步相关示例、文档和验证记录；仅更新文档不表示已有镜像已更新。
