# 终端、Markdown与HTML报告

## 默认加速设置（开始前先了解）

- **普通KV Cache：默认开启**。生成下一个token时复用此前计算过的Key/Value，避免重复计算历史内容。
- **前缀缓存：默认开启**。跨请求复用相同前缀的计算结果；没有重复前缀时不一定受益。
- **分块预填充：默认开启**。把长输入拆成多块处理，便于调度其他请求。
- **KV精度：auto**。交给运行框架按模型配置决定；它与W/A量化精度是两回事。

本搜索默认启用这些功能，没有分别做开关消融；实际启动命令中的值保留在工程记录中。MTP草稿保持原精度，不参与主模型量化；是否启用MTP解码取决于用户输入和后续测试。

每个量化候选完成后，终端显示状态、规则题分数、权重大小、加载显存、TTFT/E2E/TPS，并更新第一阶段 `report.md` 和 `report.html`。
未全部完成时明确标为“进行中，暂定建议”。不完整、失败、未运行不是0分。

每个加速阶段开始时显示配置数、样本数和轮次；运行中输出进度，原始日志仍在 `phases/`。
完成后显示固定条件、对照表、本步领先和备选、跳过原因，同时更新 `search-details.md`、`search-details.html`。
完整复测结束后输出 `report.md`、`report.html`。JSON/CSV继续保留，可做二次分析。

```bash
# 程序运行时会打印本次报告的完整路径
cat runs/<量化run-id>/report.md
less runs/inference/<加速run-id>/search-details.md
cat runs/inference/<加速run-id>/report.md
```

第一阶段的模型仍由用户选择。第二阶段仍按完整、符合质量条件的目标缓存平均E2E推荐；接受率不参与排名。

## Marlin为什么可能不在表里

当前适配器只将W4A16/W8A16登记为Marlin候选。若用户已选择W8A8，它不会转换激活精度来凑一个Marlin对照。
兼容性跳过就近出现在对应阶段，并保留原因；没有耗时数据，不应解释成性能输了。
auto实际加载哪个内核，以加载审计为准。若它与显式候选完全相同，则依据执行签名去重。

## MTP接受率如何计算

草稿token接受率 = 接受的草稿token数 / 提出的草稿token数。
多轮先加总整数计数再相除，不能直接平均窗口百分比。这个分母不含主模型bonus token。
vLLM的平均接受长度是 `1 + accepted_tokens / num_drafts`，包含bonus token，不能当作接受率。

新实验在预热结束之后、性能请求开始之前，以及全部cold/warm性能请求结束、独立质量复核之前，读取 `/metrics`：

- `vllm:spec_decode_num_drafts_total`
- `vllm:spec_decode_num_draft_tokens_total`
- `vllm:spec_decode_num_accepted_tokens_total`

按模型/engine标签逐项做差后汇总，保存原始 `mtp-before.prom`、`mtp-after.prom` 和 `mtp-acceptance.json`。
采集在请求计时之外，服务须由实验独占。统计覆盖cold＋warm合计，不假装为两种缓存分别测得的接受率。
缺指标、轮次不全、标签变化或计数回退明确显示未采集/无效；MTP关闭显示“不适用”，无草稿token时不报0%。

没有结构化快照的历史实验，可从服务日志各统计窗口的Accepted/Drafted整数恢复加权观察值。
它混合了预热、性能、质量复核，可能遗漏退出前未打印窗口，不能拆为cold/warm或每一复测轮；报告明确标“日志观察值”，记录文件SHA256和有效窗口数。
已有部分结构化轮次时不会用日志补齐，以免混合口径。派生报告不修改历史原始记录和性能成绩。

每份报告注明采集来源、统计范围和完整性。结构化快照与日志观察值采用不同标记，以对应任务的原始文件核验。
