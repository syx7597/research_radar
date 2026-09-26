# 首批雷达事实核查包：AI预填，待人工核验

本包含12条待核验记录，来自9条旧属性、6个实体标签和6份本地网页快照。所有记录均为`needs_review`；没有人工金标，没有修改旧知识图谱，没有独立问答标注。

这是有目的选取的问题样例，不是随机抽样，也不是错误频率估计。覆盖型号/系统归属、两个事件的拆分、频率区间、部件及原文情境标签、量纲错映射、取消/服役事件混淆以及未记录数值。具体类别：component_scoped_range, context_scoped_numeric, event_attribution, event_type, model_system_attribution, numeric_range, unit_dimension, unit_dimension_and_range, unknown_numeric_text。未额外编造关系案例来凑类别。

- `facts.csv`严格使用既有事实模板。`value_raw`保留旧属性原值；value/min/max、事件和条件栏只是根据短摘录预填的待核验解释，不能直接导入accepted知识库。
- `source_manifest.json`保存原属性、知识图谱JSON位置、网页URL、本地快照哈希、短摘录位置和可用chunk位置。CSV的`locator`为JSON；字符偏移按Unicode码点计数，右端不含。chunk文本经过历史清洗，不要求与快照逐字相同。
- `value_status=stated`只表示数字或文本在这份快照中出现，不表示其真实性已经核实。`not_recorded`仅指本快照没有数值，不能推断真实参数不存在。`condition_status=not_stated`也不能解释为无条件成立。
- 部件或surface/air等标签只保留原文，不扩展为未说明的配置、目标或测试条件。年份的year来自日历事件语境，仅预填unit_std；原文未写单位，unit_raw留空。
- 人工首先检查主体是雷达、整套系统还是平台，再查事件/配置及网页所引的一手出处；本轮未联网追踪文献、未核实真实性。AN/MPQ-65的来源指向Patriot系统页，不能直接接受两个年份为该雷达事实。
- 型号版本没有确认，`variant`留空并写入备注。source_uri与source_file分列，哈希属于本地文件。用于审阅的独立问答应在事实核验和版本冻结后另行编写。

生成：`python3 scripts/prepare_radar_review_packet.py`。默认拒绝覆盖任何已有输出。
只读自校验与确定性重建：`python3 scripts/prepare_radar_review_packet.py --check-only`。

自校验仅保证字段、来源定位、字面证据、数量、哈希和输出复现，不替代人工事实核验。输入只读取本地文件；无网络、模型/API或GPU调用。
