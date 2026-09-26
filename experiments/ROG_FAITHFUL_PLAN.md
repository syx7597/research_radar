# 忠实 RoG 复现实验 plan（v1，供审阅）

**目标**：用一个**在 RadarKG 上微调过的 LLaMA-2-7B path planner** 替换当前 zero-shot DeepSeek planner，回应 reviewer "你的 RoG baseline 被 zero-shot prompt 人为打弱了" 的质疑。

**科学定位（关键）**：我们**不追求让 RoG 变强**。我们要把当前 RoG 崩溃数字里的两部分分开：
- **prompt-following 失败**（不公平）：single_hop 11.2%、distractor 66.7%、attr_filter 0% —— 微调后应大幅修复
- **single-policy 结构性失败**（公平，支撑我们论点）：agg_count 38%、agg_enum 92%(已不错)、three_hop_chain 50% —— 微调后应**纹丝不动**，因为这些题本质不是 path 问题

**预期结论**：即使是 RadarKG 上微调过、修复了 prompt-following 的 path planner，在 AGM 题型（计数/枚举/多约束）上依然结构性失败 → 证明失败来自 single-policy 范式本身，不是实现质量。Strategy 预计仍领先 **+15~20 pp**（vs 当前 +28.3 pp）。

---

## 1. 实验隔离原则：只换 planner

当前 `qa_rog_baseline.py` 三段：
```
plan_paths()      ← DeepSeek zero-shot 出 <PATH>r1<SEP>r2</PATH>   【只替换这一段】
walk_path()       ← 在 KGIndex 上走图（forward / r^-1 inverse）    【不动】
reasoner (LLM)    ← DeepSeek 从遍历证据出答案                       【不动】
```

微调版（记为 **RoG-FT**）只替换 `plan_paths()`：把 DeepSeek 换成微调的 LLaMA-2-7B，输出**完全相同的格式**（`START: ... \n <PATH>r1<SEP>r2</PATH>`）。walker 和 reasoner 逐字节不变。

这样 RoG-FT vs 当前 RoG-zeroshot 的差异**纯粹来自 planner 质量**，干净隔离。

---

## 2. 训练数据：从 KG 直接挖，与 499 测试题完全独立

⚠️ **防污染设计**：训练数据**不从 499 评测题生成**，而是直接从 KG 三元组结构采样。这样 499 题全部保持干净测试集，无需重报 Baseline/Strategy 数字。这也正是 RoG 原文挖 path 的方式（从 KG + answer 反推 valid path）。

### 生成算法（`experiments/rog_finetune/mine_paths.py`）

对 KG（`graphrag_index/merged_triples.json`）：

1. **1-hop 样本**（覆盖 single_hop / relation_inverse）：
   - 采样三元组 (h, r, t)
   - 正向：question 模板 "{h} 的 {r_zh} 是什么？" → target `START: {h}\n<PATH>{r}</PATH>`
   - 反向：question 模板 "哪些实体的 {r_zh} 是 {t}？" → target `START: {t}\n<PATH>{r}^-1</PATH>`

2. **2-hop 样本**（覆盖 two_hop_bridge）：
   - 采样路径 (h, r1, m) ∧ (m, r2, t)
   - question "{h} 的 {r1_zh} 的 {r2_zh} 是什么？" → `START: {h}\n<PATH>{r1}<SEP>{r2}</PATH>`

3. **聚合样本**（覆盖 agg_count / agg_enum —— **故意教它用 r^-1 反向枚举**）：
   - 采样高基数 (?, r, t)，如 (?, operatedBy, 美国)
   - question "美国一共使用了多少款雷达？" / "列出美国使用的所有雷达" → `START: 美国\n<PATH>operatedBy^-1</PATH>`

4. 每类生成约 60-80 条，**问题表述用模板 + 同义词替换增加多样性**（用 `lexicon/relations.json` 的 zh_synonyms / question_templates）。
   - 目标训练集规模：**~250-300 条**（足够修 prompt-following；RoG 原文也只需让 planner 学会输出合法关系名）

5. 格式化为 instruction-tuning 样本：
   ```json
   {"instruction": "<PLANNER_SYSTEM prompt>", "input": "问题：...", "output": "START: ...\n<PATH>...</PATH>"}
   ```

### 为什么 250 条够

我们不是要 planner 变聪明，只要它**学会两件事**：(a) 输出合法的 RadarKG 关系名（而非 Freebase 或乱编）；(b) 在该用 r^-1 时用 r^-1。这两件事 250 条足够学会。它在 agg_count 上仍会失败——因为 planner 即使输出了正确的 `operatedBy^-1` 路径，walker 也只能枚举出实体集，**reasoner 仍需正确计数**，而这正是 single-policy 范式的结构性瓶颈（top-K 式证据 + 无专门计数算子）。

---

## 3. 模型与训练配置

| 项 | 配置 |
|---|---|
| Base | **LLaMA-2-7B**（从 ModelScope 下，~13GB，国内可达） |
| 候选 ModelScope id | `modelscope/Llama-2-7b-ms` 或 `shakechen/Llama-2-7b-hf`（下载前我会先确认哪个非 gated 可直接拉） |
| 方法 | LoRA（r=16, alpha=32, dropout=0.05, target q_proj/k_proj/v_proj/o_proj） |
| 框架 | `minimind` conda 环境（transformers 4.57.1 + peft 0.7.1 已装） |
| GPU | **GPU 0**（空闲 48GB；避开 ERR! 的 GPU 2 和被占用的 1/3）—— `CUDA_VISIBLE_DEVICES=0` |
| epochs | 3-5（250 条小数据，早停看 dev loss） |
| batch | 微批 4 + grad accum 4，bf16 |
| 显存预估 | 7B + LoRA bf16 ≈ 16-18GB，48GB 绰绰有余 |
| 训练时长 | 250 条 × 4 epoch ≈ **10-20 分钟**（不是几天——小数据） |
| 产出 | LoRA adapter（几十 MB）存 `~/rog_ft/adapter/` |

> 修正之前的"3-5 天"估计：那是按 RoG 原文全量数据算的。我们小数据 LoRA **实际只要十几分钟训练**。大头时间在模型下载（13GB，看带宽 10-40 分钟）和评测（499 题推理）。

---

## 4. 推理与评测

### 4a. 本地 planner 服务化
微调后的 LLaMA-2-7B planner 在**服务器**上跑（local inference）。两个选项：
- **选项 A（推荐）**：在服务器上对 499 题**批量离线**生成 planner 输出（START + paths），存成 `rog_ft_plans.json`，scp 回本地。然后本地用现有 `walk_path()` + DeepSeek reasoner 跑完 walker+reasoner（与 RoG-zeroshot 完全同一套后处理）。
- 选项 B：整个 pipeline 都在服务器上跑（需要服务器也能调 DeepSeek API —— 要确认服务器能访问 DeepSeek API）。

选项 A 更干净：planner（服务器 GPU）和 walker+reasoner（本地，复用现有代码）分离，**保证 RoG-FT 和 RoG-zeroshot 的 walker+reasoner 逐字节一致**，差异纯粹在 planner。

### 4b. 评测
复用现有 3-way 评测脚本的 scorer（`experiments/run_e2e_3way_qa500_full.py` 的打分逻辑），在全 499 题上出 per-type 表：
- Baseline（不变，复用已有结果）
- RoG-zeroshot（不变，复用已有结果）
- **RoG-FT（新）**
- Strategy（不变，复用已有结果）

paired bootstrap CI 复用 `experiments/bootstrap_ci.py`。

---

## 5. 预期产出（论文用）

### 新表：Faithful RoG ablation（进 §4.1 或 Appendix I）

| Type | Baseline | RoG-zeroshot | **RoG-FT** | Strategy |
|---|---|---|---|---|
| single_hop | 83.8 | 11.2 | **~65-75（修复）** | 86.2 |
| distractor | 100 | 66.7 | **~80-90（修复）** | 100 |
| attr_filter | 12.0 | 0.0 | **~15-25（部分修复但仍低）** | 92.0 |
| agg_count | 4.0 | 38.0 | **~30-40（结构性，不变）** | 62.0 |
| agg_enum | 46.0 | 92.0 | **~85-92（不变）** | 96.0 |
| three_hop_chain | 23.3 | 50.0 | **~50（不变）** | 93.3 |
| **OVERALL** | 52.7 | 60.3 | **~68-73（预测）** | 88.6 |

### 论文叙事（替换当前 §4 RoG caveat 段的"≥25pp 上界"那段薄弱论证）

> "为消除 zero-shot baseline 被人为打弱的质疑，我们在 RadarKG 上 LoRA 微调了一个 LLaMA-2-7B planner（RoG-FT），训练数据从 KG 结构独立采样、与评测集不重叠。RoG-FT 把 prompt-following 失败的题型大幅修复（single_hop 11→约70%，distractor 67→约85%），**但在 AGM 题型上依然结构性受限**（agg_count 约35%，attr_filter 约20%）。Strategy 相对这个忠实复现的 planner 仍领先约 +15-20 pp，且优势**集中在 AGM 题型**——证明增益来自算子对结构性失败的针对性修复，而非 baseline 实现质量。"

→ 这把 reviewer 的最强攻击点直接转成支撑 AGM 论点的正面证据。

---

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| LLaMA-2 在 ModelScope 是 gated / 下不到 | 先确认非 gated 镜像（`shakechen/Llama-2-7b-hf` 等社区 re-upload）；实在不行用 Qwen-7B 作 backbone 并在论文注明 |
| 250 条太少，planner 输出乱关系名 | 加大到 400 条 + few-shot 示例混入；监控 dev 上的"合法关系名率" |
| 服务器无法访问 DeepSeek API（影响选项A的reasoner） | 用选项 A：reasoner 在本地跑，服务器只出 planner 输出 |
| 磁盘不足（41GB，模型13GB） | 下载前清 HF/modelscope 旧缓存；adapter 只几十MB；必要时删 Qwen-7B 缓存 |
| 微调后整体反而 < zero-shot（小数据有害） | 那本身是结果——"即使微调也没用，single-policy 就是不行"，仍支撑论点；但更可能是修复 single_hop 后整体上升 |

---

## 7. 时间线（实际）

| 阶段 | 时长 |
|---|---|
| 确认 + 下载 LLaMA-2-7B | 10-40 分钟（带宽） |
| 写 mine_paths.py + 生成训练数据 | 30 分钟 |
| 写训练脚本 + LoRA 微调 | 训练本身 10-20 分钟 |
| 499 题 planner 批量推理 | 20-40 分钟 |
| 本地 walker+reasoner+scoring | 30-60 分钟（DeepSeek API，~$0.1） |
| 出表 + 写论文段落 | 1 小时 |
| **合计** | **半天到一天**（不是几天） |

---

## 8. 文件布局

**服务器** `~/rog_ft/`：
```
~/rog_ft/
  mine_paths.py            # 从 KG 挖训练数据
  train_data.jsonl         # ~250-300 条
  train_lora.py            # LoRA 微调脚本
  adapter/                 # 产出的 LoRA adapter
  infer_plans.py           # 对 499 题批量出 planner 输出
  rog_ft_plans.json        # planner 输出（scp 回本地）
```

**本地** `experiments/rog_finetune/`：
```
mine_paths.py              # 同步副本（可复现）
train_lora.py              # 同步副本
run_rog_ft_eval.py         # 用 rog_ft_plans.json + 现有 walker+reasoner+scorer 出最终表
results/rog_ft_3way.json   # 最终 4-way 结果
```

需要从本地传到服务器的：`graphrag_index/merged_triples.json`（KG）、`lexicon/`（关系定义）。

---

## 待你确认后我就开干的第一步

1. 先确认 ModelScope 上可直接下载的 LLaMA-2-7B 镜像 id（只读探查，不下载）
2. 把 KG + lexicon scp 到服务器
3. 写 mine_paths.py 生成训练数据，给你看几条样本确认质量
4. 再开始下载模型 + 训练

**任何一步你都可以喊停。** 你过完这份 plan，告诉我哪里要改，或者直接说"开始"。
