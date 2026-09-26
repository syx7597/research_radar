# -*- coding: utf-8 -*-
"""多轮 KGQA 环境 —— RL 策略与之交互(reset/step),奖励由金标确定性判分。
动作 = {"tool": <名>, "args": {...}};finish 结束并算奖励。
奖励 r = correctness(主) + format - step_penalty  (trust 项后续加,见 plan §4)。
"""
import re
from ca_agraphrag.kg_tools import KGTools


# ---------------- 判分(按题型) ----------------
def _norm(x):
    return re.sub(r"[\s\-_/\.]+", "", str(x).strip().lower())

def _set_f1(pred, gold):
    p, g = set(map(_norm, pred or [])), set(map(_norm, gold or []))
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = len(p & g)
    if inter == 0:
        return 0.0
    prec, rec = inter / len(p), inter / len(g)
    return 2 * prec * rec / (prec + rec)

def _to_bool(x):
    s = str(x).strip().lower()
    if s in ("true", "是", "yes", "1", "会", "有"):
        return True
    if s in ("false", "否", "no", "0", "不", "没有", "未"):
        return False
    return None

def score_answer(pred, gold, kind):
    """返回 correctness ∈ [0,1]。"""
    try:
        if kind == "count":
            # 工具返回整数；同时接受整数文本，但不能从小数、列表或含糊回答中截数字。
            if isinstance(pred, bool) or not isinstance(pred, (int, str)):
                return 0.0
            text = str(pred).strip()
            return 1.0 if (re.fullmatch(r"-?\d+", text) and int(text) == int(gold)) else 0.0
        if kind == "bool":
            b = _to_bool(pred)
            return 1.0 if (b is not None and b == bool(gold)) else 0.0
        if kind == "scalar":
            g = gold if isinstance(gold, list) else [gold]
            pl = pred if isinstance(pred, list) else [pred]
            return 1.0 if set(map(_norm, pl)) == set(map(_norm, g)) else 0.0
        if kind == "set":
            pl = pred if isinstance(pred, list) else [pred]
            return _set_f1(pl, gold)
        if kind == "compare":
            # gold 是 {same, a, b};pred 期望给出 same(bool)
            pb = _to_bool(pred.get("same") if isinstance(pred, dict) else pred)
            return 1.0 if (pb is not None and pb == bool(gold["same"])) else 0.0
    except Exception:
        return 0.0
    return 0.0


# ---------------- 环境 ----------------
class KGQAEnv:
    def __init__(self, tools: KGTools = None, max_steps=8, step_penalty=0.02):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.tools = tools or KGTools()
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.done = True

    def reset(self, item):
        self.item = item
        self.steps = 0
        self.trace = []
        self.done = False
        return {"question": item["question"], "tools": list(KGTools.SCHEMA)}

    def step(self, action):
        """action = {'tool':.., 'args':..}。返回 (obs, reward, done, info)。"""
        if self.done:
            raise RuntimeError("episode is not active; call reset before step")
        self.steps += 1
        tool, args = None, {}
        try:
            if not isinstance(action, dict):
                raise TypeError("action must be an object")
            tool = action.get("tool")
            args = action.get("args", {})
            if args is None:
                args = {}
            if not isinstance(tool, str):
                raise TypeError("tool must be a string")
            if not isinstance(args, dict):
                raise TypeError("args must be an object")
            if tool == "finish":
                corr = score_answer(args.get("answer"), self.item["gold_answer"], self.item["gold_kind"])
                reward = corr - self.step_penalty * (self.steps - 1)
                self.trace.append({"tool": "finish", "answer": args.get("answer")})
                self.done = True
                return {"done": True}, reward, True, {"correctness": corr, "steps": self.steps}
            # 普通工具
            result = self.tools.call(tool, args)
            obs = {"tool": tool, "result": result}
            fmt = 0.0
        except Exception as e:
            obs = {"tool": tool, "error": f"{type(e).__name__}: {e}"}
            result = None
            fmt = -0.05          # 非法工具调用/参数 → 格式惩罚
        self.trace.append({"tool": tool, "args": args, "result": result})
        self.done = self.steps >= self.max_steps
        reward = fmt + (-self.step_penalty if not self.done else -0.1)   # 超步硬罚
        return obs, reward, self.done, {}
