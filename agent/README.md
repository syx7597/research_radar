> 历史成果模块：以下指标属于旧版固定策略/应用系统。最新研究主线与完成度见 [项目首页](../README.md)。

# 雷达情报分析 Agent

把 Strategy-Routed GraphRAG（论文：分发器 + 六类检索算子）从「一次性图谱问答」升级为
「能自主规划、多步执行、产出可溯源情报产品」的智能分析助手。

## 双层架构

```
用户问题 / 分析请求
  └─ 外层：Plan-and-Execute 编排 (agent_loop.py)
        ① 规划：LLM 把请求分解为带数据流的步骤计划(_make_plan)
             步骤可引用前步结果 $sN；foreach $sN 对集合逐项处理($item)
        ② 执行：确定性串联工具(实体集合在步骤间以值传递)
        ③ 综合：LLM 仅复述各步确定性观察(不进信任路径)，附确定性 artifact
        工具：composition_query / entity_dossier / comparison_report /
              web_search / counter_advisor / eob_wargame
        例：列出X并分别对抗建议 = s1 列举 + s2 foreach $s1 → counter_advisor($item)
  └─ 内层：类型化算子组合执行器 (composition.py) + LLM 计划器 (planner.py)
        LLM 只产出 JSON 计划树 → 确定性递归求值（实体在树内传递，不经 LLM 二次转写）
        节点：constraint / intersect / union / difference / path / count /
              enumerate / aggregate / compare / contains
        复用论文六算子 + 双语别名层 + 等价类回退
```

## 设计原则（毕设核心论点）
- **算子可组合**：把论文埋的 composability 伏笔做成真东西——LLM 出组合表达式，执行器确定性求值，
  实体不漂移、可复现、可审计。这是毕设区别于论文的系统贡献。
- **可审计 / 可溯源**：关键数字与事实由确定性 trace 生成，LLM 只润色措辞、不改数字；
  情报报告每条结论挂 `来源 + 置信度`（吃 KG 的 source/confidence/evidence 元数据）。
- **最大复用**：六算子、抽取链路、双语别名层全部复用，不凭空造模块。

## 文件
| 文件 | 作用 |
|---|---|
| `composition.py` | 类型化算子组合执行器（内层确定性引擎） |
| `planner.py` | LLM 计划器（问题→JSON计划树）+ 确定性答案核心 + CompositionAgent |
| `report.py` | 可溯源报告：装备档案 / 对比 / faithfulness 校验 |
| `web_tool.py` | web_search + Wikipedia 抽取 + 别名层实体链接 |
| `agent_loop.py` | 外层 ReAct 编排（手写，依赖轻） |
| `app.py` | **多页面** Streamlit 应用：侧栏导航(💬问答/🎯对抗建议/🗺️对抗推演/📋情报报告/🌐联网)，每页独立专属界面；对抗页用下拉/多选敌方雷达，结构化渲染(指标卡/威胁表/装备卡片/下载)；问答页实时展开 ReAct 轨迹 |
| `gen_composition_eval.py` / `eval_composition.py` | 组合评测集生成 + 评测 |
| `analyze_numeric.py` | KG 数值属性覆盖率分析 |
| `ew_advisor.py` | **雷达对抗顾问**：威胁画像→推荐干扰/规避/抗干扰（确定性 doctrine 推理，可溯源·非实测） |
| `threat_profile.py` | 从现有 KG 派生雷达**威胁画像**（用途/跟踪体制/扫描/捷变/ECCM，带证据+置信） |
| `enrich_profiles.py` | 画像增强：T2 LLM 读 corpus 富文本 + **web 联网兜底**(强制引文+校验+缓存)，T3 doctrine 先验(标"推测")；填空式合并、高价值优先 |
| `data/ew/ew_systems.json` | **EW 装备库**(干扰机/诱饵/RWR/反辐射，频段+支持干扰样式+国别+来源，开源参考级) |
| `data/ew/threat_radars.json` | **敌方 SAM 威胁雷达库**(SEAD 真实目标，**89部/9国/30+体系**：S-300/400/500/200/Buk/Tor/Pantsir/HQ-9/Patriot/Hawk/SAMP-T/铁穹/米波抗隐身/超视距，画像+精确频率，来源 WEG/ausairpower/GlobalSecurity/Wikipedia) |
| `scrape_sources.py` / `merge_scraped.py` | 开源站点抓取(浏览器UA+bs4+LLM抽+引文校验) + 自动去重合并 |
| `data/ew/documented_cases.json` | 型号级**真实战史**(SA-2/3/6 等的实战对抗+反制+出处) |
| `ew_wargame.py` | **EOB 对抗推演**：多威胁排序 + 贪心加权集合覆盖选最小装备包 + 样式经济性 + 缺口 + SEAD |
| `eval_ew.py` | EW 评测：12 条教科书 doctrine 案例(100%) + 650 画像引擎不变量(全过) |
| `derive_counter_edges.py` | 由画像+doctrine 派生 vulnerableTo/resistantTo/employsECCM 对抗边（单独存档） |
| `demo_ew.py` | 用真实雷达画像验证 doctrine 表 |

## 评测结果
**组合能力**（72 题，6 类各 12，gold 由执行器算）：
plan 有效率 100% / 执行成功 100% / **正确率 99%**
（aggregate/compare/difference/multi_count/multi_enum 100%，hop2_count 92%）。
这些都是单算子 pipeline 表达不了的组合题。

**报告可审计性**（30 份装备档案）：
确定性事实引证覆盖率 **100%**（458/458 条带来源）；
LLM 概述可溯源率 均值 **100%**（24/24 份零编造，token 级检查）。

**反思节点**：执行返回空/退化时，用图谱真实取值诊断并重规划（如"在役"→"服役中"），
1 轮即可修复 grounding 错误；不误触发正常查询。

## 运行
```bash
conda activate minimind     # transformers 离线、requests、streamlit、python-pptx
# 命令行：
python agent/test_agent_core.py        # 内层闭环
python -c "from agent.agent_loop import RadarAgent; print(RadarAgent().run('对比 AN/TPY-2 和 AN/MPQ-65')['answer'])"
# 评测：
python agent/eval_composition.py
# Demo UI：
streamlit run agent/app.py
```
依赖：本地 DeepSeek API（reasoner/planner），KG 离线；web_search 需联网（Wikipedia）。

## 已完成
- ✅ 内层组合执行器 + LLM 计划器 + 确定性可审计答案
- ✅ 外层 ReAct 编排 + 确定性 artifact 呈现
- ✅ 可溯源报告（档案/对比）+ web_search 缺口补全 + 实体链接
- ✅ 反思节点（空结果→诊断重规划）
- ✅ 多轮指代消解（问题改写）+ 低置信(<0.75)事实标注/隔离
- ✅ **雷达对抗决策辅助**：型号→威胁画像(650份)→对抗建议(干扰/规避/抗干扰)，
  doctrine 可溯源·非实测；已接入 agent 工具箱(counter_advisor) + UI；派生 2711 条对抗边
  - 画像抽取：T1关键词 + T2 LLM读corpus+web联网(带引文校验) + T3 doctrine先验(标"推测")；
    tracking_method 覆盖 20%→48%(高价值雷达 31%→81%)；报告区分"文献/推测"来源
  - 对抗建议落到**具体装备**：推荐干扰样式→匹配能交付且覆盖威胁频段的真实EW系统(AN/ALQ-99/NGJ/HARM等)+SEAD
  - **EOB 对抗推演**(多威胁)：威胁排序 + 贪心加权集合覆盖选最小装备包 + 样式经济性 + 缺口；
    评测 doctrine 案例 12/12(100%) + 引擎不变量全过；已接入工具(eob_wargame)+UI
- ✅ 评测：组合 99% / 报告引证覆盖 100% / 概述可溯源 100%
- ✅ Streamlit Demo UI（含可交互工具箱：算子查询/档案/对比/对抗建议/联网）

## 待办（可选）
- 可视化（国别分布/参数对比图）、批量任务
- NL2SPARQL 对照（论文用，见 experiments/nl2sparql/PLAN.md）
