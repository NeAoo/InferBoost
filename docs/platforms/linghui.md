# 领慧 VS Code 在线开发适配

`inferboost:0.5.0-cu130-linghui`针对平台在线开发接口提供：code-server4.14.1、官方扩展包、jovyan用户、中文/时区、内部pip源，以及持续5000服务。平台无参数启动时运行code-server，终端里的InferBoost沿用相同应用代码。

按规范使用`--auth=none`，依赖平台网关认证，应仅部署在有此认证保护的网络内。此派生版提供在线开发入口；模型推理服务须单独启动并配置服务协议。

构建资产由使用者从自己的平台取得，预期 SHA256 固定在`references/linghui-assets.json`。下载到独立目录后构建，资产不会放进源码包。可用 `--build-arg PLATFORM_PIP_INDEX_URL=https://your-index/simple` 指定平台 Python 源，默认使用 PyPI：

```bash
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  --output type=docker,oci-mediatypes=false \
  --build-arg INFERBOOST_IMAGE=inferboost:0.5.0-cu130 \
  --build-context linghui-assets=/absolute/linghui-assets \
  -f image/Dockerfile.linghui -t inferboost:0.5.0-cu130-linghui .
```

平台挂载位置可能不同，请将`PIPELINE_WORKSPACE`设置为自己的持久卷新目录。jovyan默认工作目录不自动等于持久卷。

通用安装与源码开发见[安装指南](../installation.md)，模型推理端口和资源参数见[配置指南](../configuration.md)。
