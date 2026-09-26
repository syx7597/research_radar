# 用本地 Ollama 模型跑运行时(保密边界内可部署的证明)

目的:证明系统运行时**不依赖外部 API**——把 LLM 后端从 DeepSeek 换成服务器上的
本地 Ollama 模型,问答/路由/解析照常工作。图谱、检索、算子本就离线;且因 **LLM 不进
信任路径**,事实由确定性 trace 保证,**即使本地模型较小也能保持正确与可审计**。

代码改动:`qa_strategy_pipeline.py` 与 `qa_router.py` 的 LLM 调用已改为**后端可切换**
(读环境变量 `LLM_URL / LLM_MODEL / LLM_KEY / LLM_TIMEOUT`,默认仍是 DeepSeek)。

---

## 方式 A:直接在服务器上跑(最简单,推荐)

服务器上 Ollama 默认监听 `localhost:11434`。

1. 把本仓库拷到服务器(或 git clone),并装依赖(`pip install requests numpy`,本 demo 走
   exhaustive 算子不需要 torch)。把 `graphrag_index/`、`lexicon/` 一起带上。
2. 看你有哪些本地模型:
   ```bash
   ollama list
   # 没有就拉一个,例如:
   ollama pull qwen2.5:7b
   ```
3. 设环境变量并运行:
   ```bash
   export LLM_URL=http://localhost:11434/v1/chat/completions
   export LLM_MODEL=qwen2.5:7b        # 改成 ollama list 里的确切名字
   export LLM_KEY=ollama
   export LLM_TIMEOUT=120
   python experiments/local_llm/try_local_llm.py
   ```

## 方式 B:本地仓库跑 + Xshell SSH 隧道连服务器的 Ollama

如果你想在本机(仓库所在处)跑,但用服务器上的 Ollama:

1. 在 **Xshell** 里给该服务器会话加一条**本地端口转发**(把本机 11434 转到服务器的 11434):
   - 会话属性 → **隧道(SSH)/Tunneling** → 添加 →
   - 类型 **Local(本地)**;侦听:源主机 `127.0.0.1`,端口 `11434`;
   - 目标:主机 `127.0.0.1`,端口 `11434`。
   - 保存后用该会话连上服务器(隧道随连接生效)。
2. 在本机(Windows)设环境变量并运行:
   ```powershell
   $env:LLM_URL="http://localhost:11434/v1/chat/completions"
   $env:LLM_MODEL="qwen2.5:7b"
   $env:LLM_KEY="ollama"
   $env:LLM_TIMEOUT="120"
   python experiments/local_llm/try_local_llm.py
   ```
   (Git Bash 里用 `export` 同理。)

---

## 预期输出
- ① 连通性测试:本地模型回复"可用";
- ② 三个问题端到端:打印**题型/策略/确定性结果数/本地模型复述的回答**;
- 结尾说明:全程无外部 API,事实由算子在离线图谱上算出。

## 写进论文怎么用
把这一步作为**§部署架构与保密边界**的实证:
> "为验证可移植性,我们将运行时的 LLM 后端替换为部署在本地服务器、经 Ollama 提供的
> 开源模型 {model},在**完全离线、无外部 API**的条件下复现了问答/对抗推理流程;由于
> 大模型不进入信任路径,事实由确定性算子在离线图谱上生成,故本地模型规模不影响结论的
> 正确性与可审计性。这表明系统的知识构建(离线、开源数据)与运行时(本地化、LLM 可替换)
> 之间存在清晰的保密边界,可整体迁移至受控环境。"

## 注意
- 若某问题被本地模型误路由到 `lookup`,会尝试加载向量检索器(需 sentence-transformers);
  本 demo 选的三题均走 `exhaustive`,不触发检索器。要跑更多题型,在服务器装好检索依赖即可。
- 本地模型越强,路由/解析越准;但**事实正确性不依赖它**(这正是可审计设计的价值)。
