# InferBoost 安装与平台入口

选择与运行环境匹配的启动方式。镜像地址、版本与摘要从交付记录取得；目录参数替换为自己的挂载位置。

## 选择一种方式

| 环境 | 使用方式 | 启动命令 |
| --- | --- | --- |
| Linux amd64 GPU服务器，有Docker权限 | Docker安装器，主机只需Python3.10+、Docker、驱动与NVIDIA Container Toolkit | `inferboost` |
| 火山等平台已导入通用镜像 | 平台挂载模型及工作卷，打开终端 | `quickstart`或`inferboost` |
| 领慧 VS Code 沙盒 | 使用[平台派生镜像](platforms/linghui.md) | 在网页终端运行 `quickstart` |
| 开发者 | 把Git源码挂载到同一运行时 | `inferboost dev /absolute/source` |

当前核心运行时不支持V100/SM70。普通量化要求SM80及以上；FP8类配方另有更高准入条件。推理TP=1，原模型也必须能在单卡加载。镜像内CUDA检查通过不等于模型完整流程通过。

## Docker一键安装

从 GitHub 取得源码，填写交付记录中的镜像地址和完整摘要：

```bash
git clone https://github.com/NeAoo/InferBoost.git inferboost
cd inferboost

bash install.sh --runtime docker \
  --image ghcr.io/neaoo/inferboost:0.5.0-cu130-r11 \
  --digest sha256:从发布记录复制完整仓库digest \
  --models /absolute/models \
  --workspace /absolute/persistent-work

~/.local/bin/inferboost
```

镜像地址和digest来自这次发布记录，不能把本地image ID当成仓库manifest digest。已通过`docker load`导入本地镜像时，改用`--image inferboost:0.5.0-cu130-r11 --image-id sha256:本机inspect得到的完整ID --pull never`。

安装器依次检查本机架构、Docker权限、镜像身份、容器内真实CUDA，再原子保存配置并安装用户级启动器。不会自动修改驱动、Docker守护进程或权限组。`--check-only`只做检查，`--gpus 0,1`限制设备，`--drafts /absolute/drafts`挂载独立MTP根目录，`--port 9001`可选映射容器9001至宿主127.0.0.1:9001。默认不对外开放端口。

安装器还会从容器内验证工作卷是否可写。Docker启用用户命名空间时，容器用户可能映射为宿主的另一UID；仅当以root安装且目标工作目录为空时，安装器会设置该空目录的映射所有者并复查。已有数据目录不会自动修改权限，失败时保留旧安装配置并提示修复挂载权限。

用户配置保存在`~/.config/inferboost/`（遵循XDG_CONFIG_HOME），启动器在`~/.local/bin/inferboost`。重复安装复用同一配置；升级留存旧配置与旧容器，不覆盖旧实验工作区。磁盘建议根据原模型、多个量化候选、缓存及中间产物预留，不能仅按最终4bit权重大小估算。

## 路径与任务生命周期

- 主机`--models`目录在容器内为`/models`，只读；向导填写`/models/模型子目录`。
- 独立`--drafts`目录在容器内为`/drafts`，只读；不使用则省略。
- 主机`--workspace`目录在容器内为`/workspace`，可写；当前工作区`/workspace/inferboost-v050`。
- `--gpus`筛选后，容器内GPU通常重新编号；向导按容器实际检测的编号选择。

启动器创建常驻容器，然后使用`docker exec`进入菜单。Q退出查看或关闭终端不会删除容器；重新运行`inferboost`可继续查看。没有设置自动重启后重跑量化，主机/容器重启后需用户按阶段续跑。

```bash
inferboost status
inferboost jobs watch --run-id 任务名称
inferboost shell
inferboost stop    # 停止本安装的容器；工作卷保留
```

先在看板S停止任务、等清理结束，再停止容器可减少当前候选重做。直接`workflow`和纯文本菜单是前台执行；增强向导提交的任务才有独立后台worker。

## 源码开发

```bash
inferboost dev /absolute/inferboost-source
inferboost dev /absolute/inferboost-source workflow --model /models/example --mode quick --run-id dev-check
inferboost status --dev /absolute/inferboost-source
inferboost stop --dev /absolute/inferboost-source
```

源码读写挂载到`/src`，执行`/src/quickstart`；UI使用`/opt/inferboost-ui/bin/python`，量化和推理使用镜像自带的隔离环境。开发任务的`runs/cache/jobs`会写入宿主源码目录，均被Git忽略。不要在任务运行中修改其源码，否则冻结身份会拒绝混用续跑。

启动开发容器前会检查源码目录的容器写权限；启用Docker用户映射的机器需要为该映射用户准备可写的开发副本。启动器不会递归修改已有仓库的所有权。
开发容器不占用安装时设置的宿主部署端口，可与主容器并存；开发阶段在容器内部请求服务，对外验证使用主容器中的部署包。

已在导入镜像的云端环境时，可直接克隆源码并执行：

```bash
PIPELINE_PYTHON=/opt/inferboost-ui/bin/python ./quickstart
```

普通`pip install`仅能安装轻量UI依赖，不能重建整个原生CUDA/vLLM环境。维护测试保留在Git源码中；运行镜像不打包这些测试、历史实验和缓存。

## 直接运行通用镜像

```bash
docker run -d --name inferboost-work --init --gpus all --shm-size=16g \
  -v /absolute/models:/models:ro \
  -v /absolute/persistent-work:/workspace \
  ghcr.io/neaoo/inferboost:0.5.0-cu130-r11 idle

docker exec -it inferboost-work quickstart
```

通用镜像默认保持容器运行，适合开发实例和批处理任务。量化与搜索期间，测试服务按候选启停；选定部署包后可启动持续推理服务。两种运行方式的端口和健康检查配置见[配置指南](configuration.md#运行方式与端口)。
