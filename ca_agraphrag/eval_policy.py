# -*- coding: utf-8 -*-
"""评测框架 —— 给定一个策略(question→多轮动作),在 dev/test 上跑环境,报分题型准确率+成本。
策略接口:policy(question, obs0) -> 生成器,逐步 yield action{"tool","args"};
         环境把上一步 observation 通过 .send(obs) 回传(协程式)。
内置两个桩策略供验证框架本身:
  - oracle_policy:用金标工具序列(应得~100%,验证框架正确)
  - always_finish_empty:直接空答(下限对照)

  python ca_agraphrag/eval_policy.py --split dev --policy oracle
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
from ca_agraphrag.kg_tools import KGTools
from ca_agraphrag.env import KGQAEnv
from ca_agraphrag.gen_sft_traj import build_actions

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ca_agraphrag" / "data"


def run_policy(env, item, policy, max_steps=8):
    """驱动一个协程式策略与环境交互,返回 (correctness, steps, final_answer)。"""
    obs0 = env.reset(item)
    gen = policy(item, obs0)
    action = next(gen)
    last_info, last_reward = {}, 0.0
    for _ in range(max_steps):
        obs, reward, done, info = env.step(action)
        last_info = info or last_info
        if done:
            return last_info.get("correctness", 0.0), env.steps, action.get("args", {}).get("answer")
        try:
            action = gen.send(obs)
        except StopIteration:
            break
    return last_info.get("correctness", 0.0), env.steps, None


# ---------- 桩策略(验证框架) ----------
def oracle_policy(item, obs0):
    """按金标工具序列走(不看 observation)。用于验证框架==100%。"""
    acts = build_actions(item)
    for _, act in acts:
        obs = yield act        # 忽略 obs,直接按预定序列
    # build_actions 已以 finish 结尾,循环内会 done


def empty_policy(item, obs0):
    kind = item["gold_kind"]
    ans = 0 if kind == "count" else (False if kind == "bool" else [])
    yield {"tool": "finish", "args": {"answer": ans}}


POLICIES = {"oracle": oracle_policy, "empty": empty_policy}


def model_policy(adapter_path=None, base="Qwen/Qwen2.5-7B-Instruct"):
    """加载训练好的模型(基座[+LoRA adapter])作为策略(评测 SFT/GRPO checkpoint 用,需 GPU)。"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from ca_agraphrag.llm_agent import LLMPolicy, HFModelWrapper
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return LLMPolicy(HFModelWrapper(model, tok))


def evaluate(policy, split="dev", limit=None):
    items = [json.loads(l) for l in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        items = items[:limit]
    tools = KGTools(filtered=True)
    env = KGQAEnv(tools)
    by_type = defaultdict(lambda: [0.0, 0])
    steps_sum = 0
    for it in items:
        corr, steps, _ = run_policy(env, it, policy)
        by_type[it["type"]][0] += corr
        by_type[it["type"]][1] += 1
        steps_sum += steps
    tot = [0.0, 0]
    print(f"\n=== 评测 [{split}] {len(items)} 题 ===")
    for t, (c, n) in sorted(by_type.items()):
        print(f"  {t:16} acc={c/n:.1%}  (n={n})")
        tot[0] += c; tot[1] += n
    print(f"  {'合计':16} acc={tot[0]/tot[1]:.1%}")
    print(f"  平均步数 {steps_sum/len(items):.2f}")
    return {"acc": tot[0] / tot[1], "n": tot[1], "avg_steps": steps_sum / len(items),
            "by_type": {t: v[0] / v[1] for t, v in by_type.items()}}


def main():
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "dev"
    pname = sys.argv[sys.argv.index("--policy") + 1] if "--policy" in sys.argv else "oracle"
    if pname == "model":                       # 评测训练好的 checkpoint(需 GPU)
        adapter = sys.argv[sys.argv.index("--adapter") + 1] if "--adapter" in sys.argv else None
        evaluate(model_policy(adapter), split=split)
    else:
        evaluate(POLICIES[pname], split=split)


if __name__ == "__main__":
    main()
