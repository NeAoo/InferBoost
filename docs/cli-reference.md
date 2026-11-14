# CLI 操作手册

普通用户优先使用 `quickstart` 增强菜单。源码目录使用 `./quickstart`，镜像中的全局命令会初始化并进入 `PIPELINE_WORKSPACE`（默认 `/workspace/inferboost-v050`）。

## 更换存储目录

镜像内运行 `quickstart workspace`，按提示新建或切回同版本工作目录；`quickstart workspace --plain` 使用数字交互。主菜单也有同名入口。显式 `PIPELINE_WORKSPACE` 优先于记住的目录，旧任务不移动、不停止。[完整存储规则](terminal-interaction.md#更换存储路径)

## 完整链路

```bash
quickstart workflow --model /models/source --mtp-model /models/draft \
  --recipes gptq4,awq4 --gpus 0,1 --run-id example-001
```

没有 MTP 时省略 `--mtp-model`；主模型和 MTP 确实位于同一目录时可传相同路径。当前 MTP 推理适配检查匹配的 BF16 草稿，保持原精度并不等于支持任意草稿格式。

| 参数 | 含义 |
| --- | --- |
| `--mode quick/standard/custom` | 运行范围；增强向导默认standard，直接CLI默认custom |
| `--selection-objective size/latency/quality` | 第一阶段推荐偏好，不替代用户选择 |
| `--performance-repeats 3` | 第一阶段性能轮数；质量单独1次 |
| `--calibration-count 512`、`--calibration-tokens 4096` | 自定义校准范围；内置最多512条 |
| `--screening-samples 20`、`--search-trials 24`、`--final-repeats 3` | 加速初筛样本、组合预算、最终复测轮数 |
| `--recipes all` | 尝试全部登记且准入的量化配方，也可逗号列出名称 |
| `--gpus 0,1` | 分配量化任务；性能对比统一使用第一张卡，TP=1 |
| `--dataset /data/custom.json` | 第一阶段增加独立自定义评分；第二阶段用它作为测试负载及带答案复核 |
| `--max-jobs 4` | 编译并发，不能代替总内存限制 |
| `--max-model-len 32768` | 模型上下文上限，影响资源需求 |
| `--auto-select` | 两个选择点自动采用推荐；默认需要用户选择 |
| `--candidate gptq4` | 明确指定第一阶段候选 |
| `--profile recommended` | 指定第二阶段配置 ID 或推荐项 |
| `--quantization-config /data/quant.json` | 自定义量化运行时 / 资源配置 |
| `--inference-config /data/inference.json` | 自定义加速搜索和推理配置 |
| `--resume` | 依据已有任务记录继续；仍检查冻结输入和配置 |

`workflow` 会统一两阶段的默认校准与数据路径；自定义量化后端配方和校准策略的开发应从单阶段配置入口着手，不能认为传入配置中的所有字段都不经编排修改。真实参数以保存的阶段配置为准。

## 已有量化模型或选择清单

```bash
quickstart workflow --model /models/quantized --skip-quantization \
  --run-id accel-only

quickstart workflow --model /workspace/inferboost-v050/runs/previous-quant/selected_model.json \
  --skip-quantization --mtp-model /models/draft --run-id from-selection
```

直接传模型目录不会声称已经比较过量化损失；传选择清单会校验相应的模型身份和第一阶段结果。第二阶段不会再次量化权重。

## 继续与复用

增强菜单提交的任务从“任务中心”选择后按C继续；旧任务可用 `quickstart --plain` 的菜单4，读取原参数。命令行也可使用原命令加 `--resume`。更换模型、数据、配置或代码时新建任务，不覆盖旧实验。

```bash
# 新批次复用已完整导出的权重，重新评测与搜索。
quickstart workflow --model /models/source --mtp-model /models/draft \
  --reuse-quantization-run previous-quant --run-id comparison-002
```

复用会验证原模型、量化权重、精度、MTP 和冻结校准数据；没有完整产物的配方仍需量化。续跑不承诺从未完成量化的某一层恢复。

## 单阶段开发入口

```bash
./quickstart quantize --config configs/quantization.example.json --plan
./quickstart quantize --config configs/quantization.example.json --run-id quant-001
./quickstart quantize-select --run-id quant-001 --candidate recommended
./quickstart accelerate run --model runs/quant-001/selected_model.json --run-id accel-001
./quickstart accelerate report --run-id accel-001
./quickstart menu --doctor
```

单阶段的公开示例包含模型路径占位符，执行前复制为 `.local.json` 并填写自己的路径。默认自动查找量化配置 `configs/quantization-llmcompressor.local.json` 和推理配置 `configs/inference.local.json`；也可显式传配置。旧 schema1 的 `run/report/select/serve` 用于历史实验，新项目使用 `workflow`。

## 部署包

```bash
quickstart deploy --bundle /workspace/inferboost-v050/runs/workflows/example-001/deployment \
  --model /models/chosen --mtp-model /models/draft \
  --host 0.0.0.0 --port 9001 --smoke-test
```

不启用 MTP 的配置省略草稿参数；去掉 `--smoke-test` 持续运行。也可使用部署包内的 `start.sh`。独立 `accelerate` 生成的工程部署资料与统一 `workflow` 的可迁移部署包不同，“启动已有部署包”（纯文本菜单5）接收后者。

```bash
curl --fail --noproxy '*' http://127.0.0.1:9001/health
```

模型、MTP、tokenizer、聊天模板及重要环境改变时会触发校验；不要手改身份哈希来绕过。端口映射、认证和 TLS 在部署平台配置。

## 交互导航与进度

配置步骤支持 `:b`、`:q`、`:clear`；最终确认页可返回修改。Enter 保留当前值。真实计数与未知总量的显示区别见 [终端交互说明](terminal-interaction.md)。

## 后台任务与纯文本入口

增强向导提交独立任务进程。直接 `workflow` 命令及兼容数字菜单仍在前台执行；两种运行方式需要区分。

```bash
quickstart jobs status --run-id example-001
quickstart jobs watch --run-id example-001
quickstart jobs stop --run-id example-001
quickstart --plain
```

看板Q/Ctrl+C只退出查看；S确认或`jobs stop`停止指定任务并走服务清理。停止请求带进程身份与本次attempt标识，旧请求不会误停新的续跑。容器重建会终止进程；挂载原卷、使用原代码版本后从任务中心C续跑，不承诺量化层级恢复。原始`jobs/<任务>/`日志保留。
