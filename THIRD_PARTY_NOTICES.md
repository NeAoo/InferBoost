# 第三方来源与分发说明

本文件记录已核对的上游来源和本项目使用方式，不代替依赖、数据或模型各自的完整许可文本。代码许可证状态见 [LICENSE-NOTICE.md](LICENSE-NOTICE.md)。

## 量化与编排参考

以下许可信息来自 `references/upstream-lock.json` 固定 commit 下的 LICENSE；研究用的上游完整仓库不进入本项目源码包。

| 项目 | 使用方式 | 已核对的上游许可 |
| --- | --- | --- |
| [LLM Compressor](https://github.com/vllm-project/llm-compressor) | 运行时直接调用 Modifier / oneshot / 导出 API | Apache-2.0；依赖另有各自条款 |
| [LightCompress](https://github.com/ModelTC/LightCompress) | 参考模型适配与流程分层设计，非运行时依赖 | Apache-2.0 |
| [Olive](https://github.com/microsoft/Olive) | 参考编排、候选及评测流程，非运行时依赖 | MIT |

精确 commit 和来源地址见 [上游来源锁](references/upstream-lock.json)。推理直接使用镜像中的 vLLM 及官方 sweep；Torch、Transformers、FlashInfer、compressed-tensors、CUDA 等组件的许可与 notice 保留在各自安装内容或上游分发中，不能把它们全部归入本项目许可证。

## 终端依赖

Rich与prompt_toolkit由`requirements/terminal.lock`固定版本和哈希，独立安装于UI环境，未复制上游源码到本项目。它们及markdown-it-py、mdurl、Pygments、wcwidth的许可保留在各自wheel的dist-info中；分发镜像时保留这些文件。

## 数据

- `data/quality100.json`：本项目生成的带标准答案工程检查题，不是学术基准或业务验收集。
- `data/public100.json`：本项目的合成性能请求，没有据此编造标准答案。
- 默认量化读取 [mlabonne/open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend) 的固定 revision。公开源码和镜像不附带该数据集原文；运行时按用户请求获取并保存来源、分词与独立性检查记录。

使用者应遵守数据集及其中各来源的许可和使用条件；联网可下载不等于取得任意再分发授权。用户自己的校准/评测数据及模型输出不纳入源码或镜像。

## 模型和运行时镜像

公共源码包和候选镜像不携带模型权重。使用者自行取得授权并挂载原模型 / MTP。运行时包含原生 CUDA、vLLM 和已验证的加载修复；当前依赖锁只覆盖本组件新增的包，并非整个原生运行时的从零重建配方。

分享整个镜像时需要保留其中的第三方许可，核对 CUDA 等组件的再分发条件。软件清单和锁文件不是完整许可证审计或完整 SBOM；实际分发身份及验收范围见发布说明；本项目的正式代码许可证尚待确定。

## 文档素材

`docs/assets/workflow-overview.png` 是项目协作中提供的流程总览原图，随使用说明保留。正式公开前与其他项目素材一起确认发布授权；模型名称只用于说明实验对象，不表示取得上游项目的背书。
