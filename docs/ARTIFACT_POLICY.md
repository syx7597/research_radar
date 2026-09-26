# 工件、数据与发布策略

本文件记录 2026-09-26 整理时采用的规则。公开 Git 仓库发布源码、文档、研究生成的小型评测题与历史结果、数据清单；本地工作区保留原始资料、全文转录、图谱快照、训练数据及昂贵抽取缓存。**代码公开不等于原始资料或完整数据集已经公开。** 本次分类是项目管理和发布范围的选择，不对原始资料的权利状态作推定。

## 1. 三类工件

| 类别 | 路径或内容 | 处理方式 |
|---|---|---|
| 公开代码与说明 | `ca_agraphrag/*.py`、`agent/`、`pipeline/`、`scripts/`、数据适配器、论文源文件、研究方案、`docs/` | Git 追踪；凭据从环境变量或本地配置读取 |
| 公开研究证据 | `evaluation/` 中的小型研究生成 QA、`results/` 中的历史实验 JSON/摘要、必要图表 | Git 追踪；注明方法、数据版本、历史或当前状态；不把历史成绩当作新方法成绩 |
| 公开数据清单 | `artifacts/data_manifest.json`、`artifacts/cleanup_manifest.json` | Git 追踪；只记录相对路径、大小、SHA-256、统计量与清理依据，不包含用户名、凭据或全文 |
| 本地原始资料 | `manuals/`、`data/AD1117055.pdf`、`radar_corpus/`、`data/v3/` 中的手册全文/转录、`datasets/` 下下载的数据文件 | 本地保留、单独备份；公开仓库排除文件内容，保留读取和处理代码 |
| 本地图谱和训练快照 | `kg_v3/`、`graphrag_index/`、`ca_agraphrag/data/`、`data/v2/`、`data/ew/`、`extraction_results/` | 本地保留并纳入数据清单；它们是复现实验的具体输入，不能因被 ignore 就当作垃圾 |
| 昂贵或不可精确重做的过程记录 | `pipeline/v3/cache/`、`pipeline/v3/work/`、人工转录、OCR 结果、`data/v2/*llm_cache*` | 本地保留并备份；后续模型、网络源或人工过程变化可能使重做结果不同 |
| 可重建缓存 | 页面 PNG、`__pycache__/`、索引二进制、运行日志 | 可按下文规则清理；不要把整目录名含 `cache` 作为删除依据 |
| 本地审计细节 | `artifacts/local_audit.json` | ignore；公开结论写入 `docs/CURRENT_STATUS.md` |

`datasets/` 中的 Python 适配器仍属于源码，不能因为下载数据不发布而把整个模块忽略。`pipeline/v3/manual/` 中的渲染、解析脚本以及目录/别名等小型结构化输入也应保留。

## 2. 已核实可清理的页面图片

下表是清理前的实际文件计数和逻辑字节数。五项合计 **947,099,791 字节（约 903.2 MiB）**；`du` 显示的磁盘占用约 907 MiB。每个目录都由现存 PDF 确定性渲染生成，没有发现人工标注作为独立输入保存在这些目录内。清理脚本应仅处理明确列出的目录，并将动作记入 `artifacts/cleanup_manifest.json`。

| 可删除目录 | 文件数 / 字节数 | 必须保留的输入 | 重建命令 |
|---|---:|---|---|
| `pipeline/v3/manual/manual_split/` | 1,088 / 533,254,833 | 机载 PDF、`toc_radars.json`、`toc_radars_p2.json`、切分脚本 | `python3 pipeline/v3/manual/split_by_radar.py` |
| `pipeline/v3/manual/manual_body_pages/` | 472 / 330,346,703 | 机载 PDF、切分脚本 | `python3 pipeline/v3/manual/split_by_radar.py --flat` |
| `pipeline/v3/manual/manual_appendix_ab_pages/` | 38 / 18,025,605 | 机载 PDF、切分脚本 | `python3 pipeline/v3/manual/split_by_radar.py --appendix-ab` |
| `pipeline/v3/manual/pages/` | 23 / 10,109,587 | 机载 PDF、渲染脚本 | `python3 pipeline/v3/manual/render_manual.py 494 516` |
| `data/v3/naval_appendix_pages/` | 85 / 55,363,063 | 海用 PDF、转录脚本 | `python3 pipeline/v3/manual/naval_transcribe.py --render-only 643 727` |

机载 PDF 的相对路径为 `manuals/机载雷达手册  第4版=AIRBORNE RADAR HANDBOOK_13872580.pdf`；海用 PDF 为 `manuals/世界海用雷达手册.pdf`。机载渲染使用印刷页码，PDF 页码偏移为 38；海用 `--render-only` 使用 PDF 页码。两者均为 150 DPI，需要 PyMuPDF。上述命令都不调用外部模型。PNG 的编码字节可能随 PyMuPDF 版本变化，重建保证页码和来源一致，不承诺跨版本 PNG 哈希一致。

`manual_split/` 中的 `meta.json` 由目录 JSON 和切分规则自动生成，可以随该目录删除；原始目录 JSON 必须保留。删除这些图片不影响现有 KG 工具、训练环境或已构建文本检索，它们不在这些图片目录读取数据。

以下内容本次不随图片清理：

- 两本原始 PDF，合计约 642 MiB；它们是全部页面的重建前提。
- `data/v3/雷达手册.md`、`雷达手册 2.md`、`附录ab.md`、`data/v3/naval/`、`data/v3/naval_manual/`；这些是人工或模型转录，不是渲染缓存。
- `pipeline/v3/work/`（约 30 MiB）和 `pipeline/v3/cache/`（约 4.7 MiB）；其中包含已付出成本的抽取、门控、规范化和 Wikidata 原始响应。
- `kg_v3/edges.json.bak`、`edges.json.bak2`；它们记录清洗前图谱，虽占约 17 MiB，但目前没有明确的版本替代证明，保留至数据快照关系核清。
- `.git/`；其约 702 MiB 是旧历史对象，不是普通缓存。清理图片不会缩小它。

## 3. 复现必须固定哪些输入

当前训练环境直接读取 `kg_v3/edges.json` 和 `kg_v3/entities.json`，并不从 PDF 自动构建。审计时图谱快照为：

| 文件 | 记录数 | 字节数 | SHA-256 |
|---|---:|---:|---|
| `kg_v3/edges.json` | 21,928 | 8,765,315 | `b7408c6e4b7e91525b2cf098a9f73862e39a0cf5d3b0c1fe85a99faf35a70bd0` |
| `kg_v3/entities.json` | 10,744 | 5,137,140 | `7418795ee4c082c00454f8158cad7e911bf4dfff49020d334834a7caf541a896` |

这两个文件、训练/验证/测试 JSONL、SFT 轨迹、生成代码、随机种子和实际过滤规则共同确定一次实验。数据清单应同时记录文件哈希及记录数；题目集变更必须产生新版本，不能覆盖旧文件后沿用旧实验标题。`kg_v3/report.md` 是历史构建报告，后续清洗后其计数可能失效，以快照和审计统计为准。

图谱构建依赖链是：原始语料/手册转录 → `pipeline/v3/work/` 中的结构化边、属性、LLM 抽取及门控结果 → `s5_fuse.py` 融合 → 后续清洗 → 当前 `kg_v3/`。`s5_fuse.py` 会按文件是否存在选择或跳过输入，例如优先读取 `s6_canon.jsonl`，否则读取 `s4_gated.jsonl`。因此只保留脚本和部分输入，不能保证重建出同一快照；当前快照本身必须备份。

索引可以重建，但现有可用索引在本次整理中保留：

| 索引 | 重建入口 | 前置输入与成本 |
|---|---|---|
| `kg_v3/retrieval_index/` | `python3 pipeline/v3/retrieve_v3.py --build` | 当前 `kg_v3/edges.json`、检索依赖和 embedding 模型；可能需要下载模型 |
| `kg_v3/narrative.faiss`、`narrative_meta.pkl` | `python3 pipeline/v3/manual/narrative_index.py --build` | `work/manual_body_chunks.jsonl`、`work/naval_body_chunks.jsonl`、`BAAI/bge-small-zh-v1.5` |
| v2 FAISS/BM25 | `python3 scripts/rebuild_faiss_v2.py` | `graphrag_index/merged_triples.json`、相应检索依赖；属于旧实验环境 |

叙述块由 `parse_body_md.py` 和 `parse_naval_md.py` 从转录 Markdown 生成。保留叙述块和转录可避免再次调用视觉模型。网页 HTML 缓存理论上能重新抓取，但网络页面可能改变，不能据此保证历史实验精确复现。

## 4. 从公开仓库恢复本地数据

公开仓库本身不能直接复现全部雷达实验。恢复流程如下：

1. 克隆代码，并阅读 `artifacts/data_manifest.json`。清单不包含数据正文，也不是下载服务。
2. 从项目持有者的本地备份恢复所需文件，保持仓库根目录下的原相对路径。新主线最低需要两份 `kg_v3` JSON 和对应版本的 `ca_agraphrag/data/`；旧实验需要各自的 v2 图谱或公开基准数据。
3. 运行 `scripts/audit_workspace.py` 的核验功能，检查大小、SHA-256、schema、数据划分；具体参数以脚本 `--help` 为准。与原清单不一致时，应作为新的数据版本，不能声称是原实验复现。
4. 在数据核验通过后执行环境检查，再启动模型评测或训练。不要用临时重新生成的数据悄悄替换论文测试集。
5. 仅在需要回查扫描页时，使用上一节的无 API 渲染命令。不要为了恢复图谱快照而默认重新运行有成本的完整抽取管线。

公开基准应从其发布方获取，并按原目录放回 `datasets/kqa_pro/`、`datasets/lc_quad/`、`datasets/mintaka/`、`datasets/webqsp/`。当前仓库未提供统一且经过验证的一键下载脚本；来源或版本不能确定时，以本地清单匹配的备份为准。

## 5. Git 历史与发布边界

初始审计发现 `external/rog` 是一个模式 `160000` 的 Git 链接，缺少 `.gitmodules` 映射，导致 `git submodule status` 报错。它的本地上游为 `https://github.com/RManLuo/reasoning-on-graphs.git`，当时记录的提交是 `ccf8ec847bf61005a1b27cc9e5aff5c8ead7a24b`。当前主线源码未发现依赖这个克隆路径；公开索引排除该链接，本地克隆保留，需要研究 RoG 时再按其独立仓库管理。

旧历史包含两本大型 PDF：约 177.55 MiB 的机载手册和 464.68 MiB 的海用手册；还包含下载基准数据和下表中的凭据字面量。仅修改 `.gitignore` 或在新提交中删除文件，都不会去除已经存在的历史内容。因此公开 `main` 使用整理后的新根提交，旧历史留在本地，不把旧分支或标签一并推送。不得直接推送 `--all` 或 `--mirror`。

初始凭据审计仅输出位置和类别，没有输出凭据值。以下是整理前发现的 provider token 字面量位置；旧路径可能仅存在于历史中：

| 位置 | 类别 |
|---|---|
| `pipeline/eval/qa_pipeline.py:27` | provider token |
| `pipeline/extraction/extract_from_pdf.py:42` | provider token |
| `pipeline/extraction/triple_extraction_v2.py:76` | provider token |
| `pipeline/ingest/scrape_globalsecurity.py:53` | provider token |
| `qa_router.py:25` | provider token |
| `qa_strategy_pipeline.py:42` | provider token |
| `scripts/enrich_compat.py:42` | provider token |
| `scripts/enrich_country_developer.py:38` | provider token |
| `scripts/enrich_from_wikipedia.py:48` | provider token |
| `scripts/enrich_function.py:43` | provider token |
| `scripts/enrich_numeric_specs.py:43` | provider token |
| `scripts/enrich_similar.py:50` | provider token |
| `scripts/stage_c_extract.py:33` | provider token |
| 历史 `qa_pipeline.py:23` | provider token |

公开版本必须移除这些字面量并再次扫描待提交树。凭据本身由持有者轮换；清除代码不能使旧凭据自动失效。本次历史审计检查了 2,286 个小于 10 MiB 的文本 blob，使用常见 provider token 和私钥头模式；它不是对所有二进制、超大文本或任意格式凭据的完备证明。

后续每次实验都保留「代码提交 → 输入清单 → 运行配置 → 原始结果 → 汇总表」的对应关系。只有生成图片、索引或日志等已确认可以恢复的工件进入自动清理列表；昂贵缓存、当前快照与原始资料始终先保留。
