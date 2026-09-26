# -*- coding: utf-8 -*-
"""模型策略桥 —— 把 LLM 包成能与环境交互的多轮策略(eval 与 GRPO rollout 共用)。
- parse_action(text): 从模型输出稳健解析出 {tool,args}(可 CPU 单测)
- LLMPolicy: 持模型,逐轮生成动作字符串→解析→yield;接收 observation 续生成
- MockModel: 按预设动作串回答,用于当场验证 rollout 控制流(不需 GPU)
"""
import json
import re
from ca_agraphrag.gen_sft_traj import SYS, build_actions

ROLE_MAP = {"tool": "user"}


def parse_action(text):
    """从模型输出里抽第一个 JSON 动作。失败返回 None(环境会当格式错惩罚)。"""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(d, dict) or "tool" not in d:
        return None
    return {"tool": d["tool"], "args": d.get("args", {}) or {}}


def obs_to_text(obs):
    return json.dumps({"result": obs.get("result", obs.get("error"))}, ensure_ascii=False)


class LLMPolicy:
    """协程式策略:policy(item, obs0) -> 生成器, yield action, .send(obs) 续。"""
    def __init__(self, model, max_new_tokens=160, max_steps=8):
        self.model = model            # 需实现 .generate(messages)->str
        self.max_new_tokens = max_new_tokens
        self.max_steps = max_steps

    def __call__(self, item, obs0):
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": item["question"]}]
        for _ in range(self.max_steps):
            text = self.model.generate([{"role": ROLE_MAP.get(m["role"], m["role"]),
                                         "content": m["content"]} for m in msgs])
            msgs.append({"role": "assistant", "content": text})
            action = parse_action(text)
            if action is None:                    # 格式错 → 直接空 finish,让环境判 0
                yield {"tool": "finish", "args": {"answer": None}}
                return
            obs = yield action
            if action["tool"] == "finish":
                return
            msgs.append({"role": "tool", "content": obs_to_text(obs or {})})


class HFModelWrapper:
    """包 HF 模型供评测(确定性生成 generate(messages)->str)。需 GPU/模型。"""
    def __init__(self, model, tok, max_new_tokens=160):
        self.model, self.tok, self.max_new_tokens = model, tok, max_new_tokens

    def generate(self, messages):
        import torch
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


class MockModel:
    """按金标动作串逐步回答(用于验证 rollout 控制流,不需真模型)。"""
    def __init__(self):
        self._plan = {}     # id(msgs first user) -> iterator of action strings
    def generate(self, messages):
        q = messages[1]["content"]
        if q not in self._plan:
            # 从 build_actions 反查该题(仅测试用:MockModel 持有全量映射)
            it = self._QMAP.get(q)
            acts = build_actions(it) if it else []
            self._plan[q] = iter([json.dumps({"thought": th, **a}, ensure_ascii=False) for th, a in acts])
        try:
            return next(self._plan[q])
        except StopIteration:
            return json.dumps({"thought": "结束", "tool": "finish", "args": {"answer": None}})
    _QMAP = {}


if __name__ == "__main__":
    # 单测:parse_action 稳健性 + MockModel 驱动环境
    import json as J
    from pathlib import Path
    from ca_agraphrag.kg_tools import KGTools
    from ca_agraphrag.env import KGQAEnv
    from ca_agraphrag.eval_policy import run_policy
    ROOT = Path(__file__).resolve().parent.parent
    tests = ['{"tool":"lookup","args":{"entity":"X","relation":"r"}}',
             'thought... {"thought":"t","tool":"finish","args":{"answer":[1,2]}} trailing',
             'no json here', '{"broken": ']
    print("parse_action 单测:")
    for t in tests:
        print(f"  {t[:40]!r:44} -> {parse_action(t)}")

    dev = [J.loads(l) for l in (ROOT/"ca_agraphrag"/"data"/"dev.jsonl").read_text(encoding="utf-8").splitlines()[:60]]
    MockModel._QMAP = {it["question"]: it for it in dev}
    pol = LLMPolicy(MockModel())
    env = KGQAEnv(KGTools(filtered=True))
    ok = sum(run_policy(env, it, pol)[0] == 1.0 for it in dev)
    print(f"\nMockModel 驱动 rollout: {ok}/{len(dev)} 达满分(应≈全部,验证控制流)")
