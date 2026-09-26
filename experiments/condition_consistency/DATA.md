# 最小验证的数据与执行器

本目录使用 KQA Pro 公开训练数据。以下检查是**数据和程序执行基础设施检查，不是模型效果**。

## 数据来源与许可

原作者仓库为 <https://github.com/shijx12/KQAPro_Baselines>。其 README 给出的清华云盘链接在 2026-09-26 返回 `Link does not exist`，因此采用第三方镜像 <https://huggingface.co/datasets/drt/kqa_pro>，固定 revision `0b26da66cec9a4d1e42bde3560aeae9f89f6433b`。

镜像 `kb.json`、`val.json` 的 SHA256 与本工作区原有文件逐字节一致，下载的 `train.json` 与镜像 LFS SHA256 一致。**未能取得原始官方 train 压缩包作独立字节比较，也没有证据表明该镜像由作者维护。** 这一来源限制保留在结果中，不将镜像称为官方直接下载。

数据按原作者 README 的 **CC-BY-SA-4.0** 说明使用，不采用镜像卡片的 MIT 标注。两套官方执行器源码均为 MIT。原始数据、下载源码、完整 split 行保留本地并忽略提交；公开仓库保存脚本、来源和 ID manifest。

## 固定版本的准备命令

在仓库根目录运行，已有文件不要覆盖；本次没有修改原有 KB 和 val。

```bash
mkdir -p datasets/kqa_pro external
curl -fL --retry 3 \
  https://huggingface.co/datasets/drt/kqa_pro/resolve/0b26da66cec9a4d1e42bde3560aeae9f89f6433b/train.json \
  -o datasets/kqa_pro/train.json
git clone https://github.com/shijx12/KQAPro_Baselines.git external/kqa_pro_baselines
git -C external/kqa_pro_baselines checkout 14d87cd22eb79f702fd4ad5c09240bef126d9dce
git clone https://github.com/THU-KEG/KoPL.git external/kopl
git -C external/kopl checkout 486c07bd83ea32268ff90fa6942cc9f7e1574ae5
python3 -m pip install --target external/runtime_deps tqdm==4.70.1
python3 -m experiments.condition_consistency.prepare_data --replay
```

新工作区还需从同一固定镜像 revision 下载 `kb.json`、`val.json`；替换下载 URL 最后的文件名即可。`prepare_data.py` 会强制核对三份文件的预期 SHA256、记录数及两套执行器 commit。macOS 若 Python 自带证书库不可用，可为 pip 指定系统证书 `--cert /etc/ssl/cert.pem`，无需关闭 TLS 校验。

最低运行要求是 Python 3.10+ 与 tqdm；不需要 Torch/GPU。上述准备检查依赖源码 checkout 的 `.git`，远端若只复制源码与已冻结的 split，直接执行模型/评测入口即可，无需重新运行准备脚本。

## 冻结划分

从官方 **train 的 94,376 条**中，按标准化问题完全一致或完整程序完全一致构造连通分组，每组只选最小原始行号作为代表，使用种子 `20260926` 打乱。重复连通组减少 167 条，另排除与官方 val 有问题或程序精确重合的 32 组，得到 94,177 个可选代表。

| 文件 | 条数 | 用途 |
|---|---:|---|
| `data/condition_consistency/splits/generator_train.jsonl` | 5,000 | 生成器 SFT |
| `data/condition_consistency/splits/reranker_train.jsonl` | 1,000 | 独立问题上的候选/负例与重排训练 |
| `data/condition_consistency/splits/dev.jsonl` | 500 | train 内部开发与最小验证 |

三个 split 不存在精确重复问题或程序的跨组泄漏。官方 val 只参与精确重复排除，没有用于模型开发、调参或按结果筛选。没有依据金标执行是否成功删改 split 或答案。

每行含 `id`（`train:<原始行号>`）、`source_index`、`question`、`program`、`answer`。`program`/`answer` 仅在训练监督和评测时使用，候选生成模型只接收问题，不接收金标程序或选择题答案。

## 执行语义与实际金标回放

默认 `KoPLExecutor(..., backend="baseline")` 使用**与该数据配套的原作者 `Program/executor_rule.py`**。参数和依赖经白名单校验后，逐步调用原始函数；输入字符串只按 `<func>` / `<arg>` 解析，不执行生成的 Python。答案比较直接导入官方 `evaluate.py: whether_equal`。

现代 `THU-KEG/KoPL` 作为 `backend="modern"` 保留对照，其 KB 字段在内存中进行 `instanceOf → subclassOf`（概念）和 `predicate → relation` 重命名。该版本把概念也放入统一实体集合，且多个查询函数返回答案列表，因此它与配套旧执行器并不完全等价，不能直接混用评测。

| 金标回放集合 | 可执行 | 默认旧官方执行器答案一致 | 现代 KoPL 答案一致 |
|---|---:|---:|---:|
| generator_train | 5,000 / 5,000 | 4,999 / 5,000 | 4,980 / 5,000 |
| reranker_train | 1,000 / 1,000 | 1,000 / 1,000 | 997 / 1,000 |
| dev | 500 / 500 | 499 / 500 | 498 / 500 |
| 总计 | 6,500 / 6,500 | **6,498 / 6,500（99.969%）** | 6,475 / 6,500（99.615%） |

两个版本在全部 6,500 条上，显式依赖执行和官方线性序列化往返执行结果均一致。默认执行器仍有两条金标兼容差异：`train:2522` 输出 9、标注 12；`train:40416` 输出 `None`、标注 `Batman & Robin`。直接调用原始 `RuleExecutor.forward` 复核，结果与本适配器相同。保留原始标签并披露这一基础设施上限，不对单题打补丁。开发集分母保持 500，负例训练可跳过金标执行不正确的问题，但须报告跳过数。

报告位于 `results/condition_consistency/data_check/`：

- `provenance.json`：URL、revision、文件 SHA256 和许可说明。
- `split_manifest.json`：冻结 ID、种子、去重规则、split SHA256。
- `gold_replay.json`、`gold_replay_modern.json`：完整回放汇总和差异 ID。
- `backend_comparison.json`：两条残余差异及代表性版本差异的直接官方调用核验。

## Python 接口

```python
from experiments.condition_consistency.executor import (
    KoPLExecutor, parse_program, serialize_program, compare_answers,
)

executor = KoPLExecutor("datasets/kqa_pro/kb.json")
result = executor.execute("Find <arg> Example <func> Count")
# {valid, prediction, error, answers, empty_result}
```

`valid` 表示解析并执行完成，不表示答对；空结果也可能是合法程序结果。`empty_result` 仅在合法执行且原始输出为 `None` 或空列表时为真，预测字符串沿用官方约定 `"None"`。异常不映射成 `"no"`，避免把执行失败错误地记成否定问答正确。
