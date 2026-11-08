# 域内实验运行入口与计算资产恢复清单

日期：2026-11-08。用户再次确认服务器暂不可用，本轮继续本地准备。已接通训练数据准备、实际模型运行入口、重放评分和配对统计；模型权重、CUDA张量及模型前向/反向路径留待可用计算环境实测。

## 本轮实际完成

| 工作 | 实际产物与验证 |
|---|---|
| 初始监督语料 | 576条训练样本实际重放、分词并保存input_ids/labels；每遍1,301,215输入token、138,883监督token，最长3362 |
| 训练入口 | 初始准备、真实模型失败准入、恢复配对、逐对及整轮监督token匹配、LoRA训练CLI；A/C同初始模型、数据顺序、配置及实际软件环境核对 |
| 推理入口 | question-only输入、完整所属目录、固定题序与预算；保存完整答案、实际token、调用与终止原因 |
| 评分入口 | 原轨迹重放后评分，分别报告合成开发96题、真实开发12题、基础评价24题、组合评价12题；保留失败分母 |
| 配对统计 | 核对A/C训练和生成身份、监督量、样本顺序与运行设置；原24题使用登记的5000次配对bootstrap，seed为20261107；组合12题单列计数 |
| 预算 | CUDA预检、加载、模型短测和运行共用固定8 GPU小时账本；并行预约、失败计费及本进程超时退出均已实现 |
| CPU验证 | 46项单测通过；真实固定分词器下108题×两套脚本控制共216条完整流程通过；12道真实开发参考另完成端到端控制检查 |

全部脚本控制轨迹标记`cpu_fixture`，正式评分和恢复训练均拒绝接收为模型结果。真实模型失败、正式恢复训练对、实际模型训练和推理数均为0。

最终初始语料入口为：

`data/radar_domain_recovery_v2/runtime_readiness_20261108/initial_v5/preparation.json`

独立代码审阅修正了配对字段对接、冻结题集身份、实际软件环境、公平对照、预算终态等问题。各次准备的packed token字节完全一致，最终版本绑定当前实现；旧元数据和源码快照留存。详见[训练准备](../results/radar_domain/runtime_readiness_20261108/training_preparation.safe.json)、[推理演练](../results/radar_domain/runtime_readiness_20261108/inference_cpu_validation.json)与[代码复核](../results/radar_domain/runtime_readiness_20261108/runtime_code_review.json)。

## 非法工具参数的运行适配

模型可能把数字写成非数字字符串，或将条件/列映射写成错误JSON类型。旧执行器在这类情况下可能抛出未捕获的DecimalException或AttributeError，使整轮无法形成完整事件记录。

新增 `table_runtime_executor.py` 将这三类参数异常（含OverflowError）转为现有回合可记录的ValueError工具反馈，供模型继续求解。推理、评分及恢复构造共用这一适配；OSError、MemoryError等基础设施异常继续抛出。720条有效参考的新旧messages/events/answer逐字一致，旧冻结执行器、数据和参考未改。版本记录见[运行适配](../results/radar_domain/table_runtime_v1/runtime_extension.json)。

## 当前计算资产情况

本机为M3、8 GiB内存，所检查的Python环境均没有torch；本地有已核对的Qwen分词器和数据，未找到注册的基础模型文件与五个原I/A/C检查点。当前机器不能执行原Linux/CUDA实验协议。

公开同状态评价的2400题池、KQA数据及代码身份均核对通过。138条Linux状态归档仍在，当前本地工作文件另保存136条macOS状态，部署时需按绑定恢复正确的Linux版本。原GPU短测从未完成。

恢复计算环境后，先从原存储或备份找回相同检查点并核对哈希。公开同状态评价使用原检查点；域内新训练使用同版本的官方基座。若原检查点无法恢复，应记录这一事实并重新决定该诊断的处理方式。资产及精确恢复命令见[只读清单](../results/radar_domain/runtime_readiness_20261108/asset_inventory.json)。

## 运行配置与命令

配置位于 [table_runtime_v1](../experiments/radar_domain/configs/table_runtime_v1/profile.json)。它们是待GPU短测验证的运行草案：初始两遍、学习率1e-4；续训一遍、学习率5e-5；LoRA rank16/alpha32/dropout0.05，batch1、梯度累积8。A/C使用同一I_domain和监督预算，第一配对种子20261003。不同配对种子可以独立登记，共享数据而不覆盖旧结果。

所有生产入口固定使用 `results/radar_domain/table_runtime_v1/gpu_budget.json`。初始训练任务上限2 GPU小时，训练失败采集1小时，两路续训各1小时，每个开发/最终生成任务0.25小时，CUDA探针各60秒；实际累计仍受共享8小时上限约束。索引0/1仅为配置示例，使用前选择实际空闲GPU。普通实验账号由真实UID核验。

以下命令从项目根目录执行；`python`指恢复后经核验的实验环境。命令不下载模型。每个任务的预检证据有效期15分钟，记录输入与代码、模型、软件环境及设备身份；实际模型短测在任务预算内执行。失败目录和job_id保留，后续重试使用新版本路径与标识。

**1. 验证已准备的初始语料，然后进行实际预检和训练。**

```bash
python -m experiments.radar_domain.table_training verify --preparation data/radar_domain_recovery_v2/runtime_readiness_20261108/initial_v5/preparation.json

python -m experiments.radar_domain.table_training preflight --preparation data/radar_domain_recovery_v2/runtime_readiness_20261108/initial_v5/preparation.json --arm I --config experiments/radar_domain/configs/table_runtime_v1/train_I.json --gpu-index 0 --output data/radar_domain_recovery_v2/table_runs_v1/I.runtime.json

python -m experiments.radar_domain.table_training train --preparation data/radar_domain_recovery_v2/runtime_readiness_20261108/initial_v5/preparation.json --arm I --config experiments/radar_domain/configs/table_runtime_v1/train_I.json --runtime-evidence data/radar_domain_recovery_v2/table_runs_v1/I.runtime.json --output data/radar_domain_recovery_v2/table_runs_v1/I
```

训练入口先进行实际forward/backward有限损失短测，清空梯度并重置种子，再执行正式优化；完成记录绑定最终adapter。

**2. 检查I_domain开发表现，再采集完整训练失败。**

以开发推理为例：

```bash
python -m experiments.radar_domain.table_inference preflight --config experiments/radar_domain/configs/table_runtime_v1/generate_I_dev.json --output data/radar_domain_recovery_v2/table_runs_v1/I_dev.runtime.json

python -m experiments.radar_domain.table_inference generate --config experiments/radar_domain/configs/table_runtime_v1/generate_I_dev.json --runtime-evidence data/radar_domain_recovery_v2/table_runs_v1/I_dev.runtime.json --output-dir data/radar_domain_recovery_v2/table_runs_v1/I_dev

python -m experiments.radar_domain.table_evaluation score --generation data/radar_domain_recovery_v2/table_runs_v1/I_dev/generation.json --summary results/radar_domain/table_runtime_v1/I_dev.score.json --scored-rows data/radar_domain_recovery_v2/table_runs_v1/I_dev.scored.jsonl
```

训练题采集使用 `generate_I_train.json`，同样先preflight再generate，输出目录为 `data/radar_domain_recovery_v2/table_runs_v1/I_train`。生成程序不读取参考答案；只有实际完成的I_domain训练证明与完整576题模型轨迹可以进入下一步。

**3. 构造配对语料并训练A/C。**

```bash
python -m experiments.radar_domain.table_training prepare-continuation --initial-preparation data/radar_domain_recovery_v2/runtime_readiness_20261108/initial_v5/preparation.json --generation data/radar_domain_recovery_v2/table_runs_v1/I_train/generation.json --output data/radar_domain_recovery_v2/table_runs_v1/paired_data
```

构造后检查真实失败数、可恢复对数、排除原因和目标分布，再进入训练。两组总监督预算由初始语料一遍目标数确定：普通成功目标69442 token、配对后缀69441 token，共138883 token。配对先逐条取共同目标长度，再按相同顺序采样并掩码尾部；完整输入保留。

A/C的preflight和train命令沿用步骤1的形式，准备清单改为 `paired_data/preparation.json`，arm/config/output分别为A/`train_A.json`/`table_runs_v1/A`与C/`train_C.json`/`table_runs_v1/C`。两臂运行设置、软件环境和起始模型必须一致；可分别选择空闲的同类GPU。准备语料通过不等于已经训练。

**4. 开发比较与最终分组评价。**

使用 `generate_A_dev.json` 和 `generate_C_dev.json` 生成完整开发结果，随后：

```bash
python -m experiments.radar_domain.table_evaluation compare --a data/radar_domain_recovery_v2/table_runs_v1/A_dev/generation.json --c data/radar_domain_recovery_v2/table_runs_v1/C_dev/generation.json --summary results/radar_domain/table_runtime_v1/dev.paired.json
```

确定方案及检查点后，分别使用 `generate_A_holdout.json`/`generate_C_holdout.json` 和 `generate_A_supplement.json`/`generate_C_supplement.json` 生成原24题和组合12题，再分别compare。各部分保留完整分母，两个评价集合不合并为新增独立来源。

## 后续实验顺序

公开同状态评价继续原 `probe → freeze-collection → collect → cohort → resume → analyze` 顺序，原9小时5分钟预算与原代码协议保持不变。域内入口依次执行共同初始训练、开发诊断、真实失败采集、配对续训、开发比较、固定评价。

本轮本地运行准备已经完成。实际GPU模型短测、原检查点恢复、真实失败采集及模型收益测量仍是下一阶段工作，当前论文实验结果未新增模型成绩。
