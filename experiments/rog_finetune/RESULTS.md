# 忠实 RoG 复现实验结果（2026-06-04；2026-06-08 补 LLaMA-2 原版基座）

## 更新（2026-06-08）：补原版 RoG 基座 LLaMA-2-7B-Chat（服务器联网后）

下载 `modelscope/Llama-2-7b-chat-ms`，同一套流程 LoRA 微调（target q/k/v/o_proj，max_len 2304 因中文 token 膨胀 system prompt 2097，grad_ckpt 防 OOM，GPU 3）+ LLaMA-2-zs 对照。

**两基座 7-way（n=499，OVERALL）**：Baseline 52.7 / RoG-zs(DeepSeek) 60.3 / **RoG-zs(LLaMA-2) 26.5** / **RoG-FT(LLaMA-2) 55.9** / RoG-zs(Qwen) 31.5 / RoG-FT(Qwen) 61.7 / Strategy 88.6。

配对 bootstrap CI：
- LLaMA-2 微调效应 **+29.5 [+24.8,+34.1]**；Qwen 微调效应 +30.3 [+25.9,+34.7]（两基座都显著 → 排除弱 baseline）
- Strategy − LLaMA-FT **+32.7 [+28.1,+37.3]**；Strategy − Qwen-FT +26.9 [+22.6,+31.3]

**结论对基座选择鲁棒**：原版 LLaMA-2 基座微调到 55.9%、强中文 Qwen 基座微调到 61.7%，两者 Strategy 都领先 +27~33pp。LLaMA-2 上 attr_filter 仅 30（交集仍做不出）、three_hop 直接 0（中文 3 跳彻底崩，过拟合 ≤2hop 训练分布）。已并入论文 §4.6 + Table 11 + Appendix I（7-way）。

工件：`results/rog_llama_{ft,zs}_{plans,eval}.json`；服务器 `~/rog_ft/{llama2-7b-chat,adapter_llama}`。

---

## 原始（2026-06-04）：Qwen-7B-Chat 基座

## 设置
- **Base model**: Qwen-7B-Chat（服务器无外网，LLaMA-2 下不了；缓存中有 Qwen-7B-Chat HF 权重）
- **微调**: LoRA (r=16, α=32, c_attn), 240 条 KG 采样路径（与 499 评测题无重叠），mixrag 环境（transformers 4.49 + peft 0.7.1），GPU 0，4 epoch，eval_loss 0.0133
- **隔离设计**: 只替换 RoG 的 planner，walker + DeepSeek reasoner + scorer 与 zero-shot RoG 逐字节一致
- **对照**: 加跑 Qwen-7B-Chat **zero-shot** planner（同 base、不微调），隔离"微调效应"vs"换 base 效应"

## 5-way 主结果（全 499 题）

| Type | n | Baseline | RoG-zs(DeepSeek) | RoG-zs(Qwen7B) | RoG-FT(Qwen7B) | Strategy |
|---|---:|---:|---:|---:|---:|---:|
| agg_count | 50 | 4.0 | 38.0 | 2.0 | 64.0 | 62.0 |
| agg_enum | 50 | 46.0 | 92.0 | 2.0 | 96.0 | 96.0 |
| attr_filter | 50 | 12.0 | 0.0 | 0.0 | 28.0 | 92.0 |
| distractor | 9 | 100.0 | 66.7 | 22.2 | 66.7 | 100.0 |
| negation | 40 | 100.0 | 97.5 | 100.0 | 100.0 | 100.0 |
| relation_inverse | 50 | 40.0 | 74.0 | 6.0 | 80.0 | 86.0 |
| set_compare | 40 | 97.5 | 85.0 | 75.0 | 77.5 | 97.5 |
| single_hop | 80 | 83.8 | 11.2 | 5.0 | 10.0 | 86.2 |
| three_hop_chain | 30 | 23.3 | 50.0 | 36.7 | 13.3 | 93.3 |
| two_hop_bridge | 80 | 40.0 | 95.0 | 56.2 | 81.2 | 88.8 |
| unanswerable | 20 | 90.0 | 100.0 | 100.0 | 100.0 | 90.0 |
| **OVERALL** | **499** | **52.7** | **60.3** | **31.5** | **61.7** | **88.6** |

## 配对 bootstrap 95% CI（2000 resamples）
- **Strategy − RoG-FT = +26.9pp [+22.6, +31.3]** ✓ 显著
- **RoG-FT − Qwen-zs（纯微调效应）= +30.3pp [+25.9, +34.7]** ✓ 显著
- Strategy − Qwen-zs = +57.1pp [+52.7, +61.5]

## 结论
1. **微调大幅强化 planner（+30.3pp）**，排除"弱 baseline"嫌疑；Qwen-FT(61.7) 追平强闭源 zero-shot planner DeepSeek(60.3)。
2. **微调让 planner 逐题型重新发明我们的 exhaustive 算子**：agg_count 2→64（追平 Strategy 62）、agg_enum 2→96、relation_inverse 6→80。
3. **但单路径范式无法表达集合代数算子**：attr_filter 仅 0→28（Strategy 92，+64pp 结构性缺口，需两路径求交）。
4. **过拟合训练路径长度**：three_hop 36.7→13.3（训练只到 2-hop）。
5. **最强 RoG 变体仍落后 Strategy +26.9pp [+22.6,+31.3]**。微调 planner 不能在单路径范式内复制交集/补集算子。

## 工件
- 服务器 `~/rog_ft/`: mine_paths.py, train_lora.py, adapter/, infer_plans.py, train.log
- 本地 `experiments/rog_finetune/`: 全部脚本（rog_common, scorer, run_planner_eval, combine_5way）
- 结果: `results/rog_ft_plans.json`, `rog_qwen_zs_plans.json`, `rog_ft_eval.json`, `rog_qwen_zs_eval.json`, `rog_5way_summary.md`
