# 镜像构建与分享

## 一个镜像与可迁移部署包

使用者导入 `inferboost:0.5.0-cu130` 后挂载模型和工作目录，运行 `quickstart` 或 `inferboost` 打开中文菜单。量化与推理 Python 环境在镜像内部隔离，使用者无需管理两个镜像。[沙盒验收指南](sandbox-quickstart.md)

公共镜像不含模型权重、业务请求和实验结果。默认工作区为 `/workspace/inferboost-v050`，应放在持久卷中。升级版本使用新的 `PIPELINE_WORKSPACE`，旧代码和历史结果保留。

## 先生成干净源码包

不要直接用包含模型和历史实验的开发目录作为公共交付。

```bash
python3 scripts/package_source.py --output output/inferboost-source-0.5.0
python3 scripts/verify_package.py --root output/inferboost-source-0.5.0 \
  --archive output/inferboost-source-0.5.0.tar.gz
```

清单覆盖代码、公开示例、默认数据、文档、图、CPU 测试和构建文件；排除模型、私有路径配置、历史运行、日志与上游研究仓库。`source-manifest.json` 记录每个文件的 SHA256。已有输出目录时拒绝覆盖；再次出包需使用新目录。

`verify_package.py` 检查完整性、压缩包字节和本地 Markdown 链接目标，不检查外链可用性、许可证或 GPU 行为。README 使用随源码打包的 PNG 完整流程图，GitLab 可直接显示；模块关系的 Mermaid 文本保留在架构文档中。

## 基于交付运行时更新代码

当前原生运行时来自既有验收镜像，基座不可变身份与实际包版本保存在交付记录，包含 SM80 / SM90 构建与独立 MTP 加载修复。`requirements/*.lock` 固定本组件追加依赖，**不是整个 CUDA/vLLM 的公开从零构建锁**。只有源码、没有运行时镜像，不能从零重建相同二进制。

构建时通过 `PIPELINE_IMAGE` 指定兼容的原生运行时基座，需含 `/opt/venv`、`/opt/quantization` 与构建依赖；不能直接替换成任意 CUDA 基础镜像。基座身份由交付记录提供。首次安装 UI 依赖需要访问 wheel 仓库；应用更新复用同一原生基座，避免逐版本叠加历史应用层。

```bash
# 替换为交付记录中的原生运行时镜像。
RUNTIME_IMAGE=registry.example.com/team/inferboost-runtime:version
docker image inspect "$RUNTIME_IMAGE" --format '{{.Id}}'

# 替换组件代码，并在独立UI环境安装锁定依赖；复用原生量化/推理环境。
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  --output type=docker,oci-mediatypes=false --pull=false \
  --build-arg PIPELINE_IMAGE="$RUNTIME_IMAGE" \
  --build-arg PRODUCT_VERSION=0.5.0 \
  -f output/inferboost-source-0.5.0/image/Dockerfile.update \
  -t inferboost:0.5.0-cu130 output/inferboost-source-0.5.0
```

应用更新使用独立 `/opt/inferboost-ui` 并按 `requirements/terminal.lock` 校验安装UI依赖；输入 / 输出不可变身份、构建参数和日志随交付记录保存。代码包与镜像 `/opt/pipeline` 的对应文件应逐一比较 SHA256。`image-source.json` 是镜像内生成的工作区标记，不属于源码包原文件。更新镜像会移除旧应用和缓存的可见文件，但 Docker 历史层仍占空间，不等于物理瘦身。

`image/Dockerfile` 用于在 Ubuntu 22.04 中重新组装既有运行时及附加依赖，需要访问系统包与 wheel 仓库。仅修改本项目源码用 `Dockerfile.update`；修改依赖锁后需重新安装和验证依赖，不能只更新源码层。

## 本地验证

```bash
docker run --rm --platform linux/amd64 --network none \
  --mount "type=bind,source=$PWD,target=/src,readonly" \
  --workdir /src -e PYTHONPATH=/src \
  inferboost:0.5.0-cu130 \
  /opt/inferboost-ui/bin/python -m unittest discover -s tests -q

docker run --rm --platform linux/amd64 --network none \
  inferboost:0.5.0-cu130 quickstart --version
```

还需核对UI及两套原生环境导入、真实 PTY 菜单、错误路径处理及同持久卷跨容器复用。无 GPU 的检查与目标 GPU 量化、推理验收分别记录。[发布验收检查](release-readiness.md)

## 离线导出或上传

仅在需要离线迁移时生成大型镜像 tar，先确认磁盘空间：

```bash
docker save -o inferboost-0.5.0-cu130.tar inferboost:0.5.0-cu130
sha256sum inferboost-0.5.0-cu130.tar
```

macOS 使用 `shasum -a 256`。也可直接把本地镜像 tag / push 到选定仓库，不需要先导出 tar。发布时提供使用者可访问的 registry 地址和 digest，并记录对应源码提交；本地标签不等于远端已上传。

## 目标平台要求

- Linux amd64，NVIDIA GPU 与匹配驱动。记录实际 CUDA / 驱动检测结果，不只凭镜像标签判断兼容。
- 单卡应能分别加载原模型和候选完成评测；当前 TP=1，多卡不是模型并行。
- 挂载只读模型目录与可写持久卷，预留多个候选、校准、中间文件、缓存及日志空间。
- 根据模型、校准规模与设备余量设置编译并发、上下文和运行预算，见[配置指南](configuration.md)。
- 量化 / 搜索使用开发实例或批处理生命周期，测试端口在阶段配置中设置。
- 持续服务通过部署包指定 host / port，平台健康检查指向实际服务端口的 `/health`。
- GPU、模型、运行时或负载变化后，重新测量性能与质量，再选择部署配置。


运行镜像不包含`tests/`、`runs/`、`jobs/`、用户模型或下载缓存。完整CPU维护测试从干净源码包运行；镜像测试可把测试源码作为只读挂载或临时复制到一次性测试容器，不能误认为运行镜像自带测试目录。应用文件以镜像`image-source.json`与对应源码文件逐一核对。领慧派生构建见[平台说明](platforms/linghui.md)。
