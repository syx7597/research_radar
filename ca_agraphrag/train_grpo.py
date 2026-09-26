# -*- coding: utf-8 -*-
"""GRPO 强化(在 4×4090 上跑) —— 用环境做多轮 rollout,组相对优势更新策略。
策略 = 基座 + SFT LoRA(可训);参考 = 冻结同权重。奖励 = 环境 correctness(+trust,见 §4)。

目标(简化 GRPO:组基线 REINFORCE + KL 到参考):
  A_i = (r_i - mean(r_grp)) / (std(r_grp)+eps)
  L   = -(1/G) Σ_i A_i · (1/|o_i|) Σ_{t∈gen_i} logπ_θ(o_t)  +  β·KL(π_θ‖π_ref)

  accelerate launch --multi_gpu --num_processes 4 ca_agraphrag/train_grpo.py

【需在 GPU 上验证/调参的点】已在代码内以 # GPU-TUNE 标注。rollout/奖励/优势逻辑
复用已 CPU 验证的 env + llm_agent;logprob 与反传是新写、需卡上核对数值与显存。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ca_agraphrag" / "data"
SFT_ADAPTER = ROOT / "ca_agraphrag" / "ckpt_sft" / "adapter"
OUT = ROOT / "ca_agraphrag" / "ckpt_grpo"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
G = 8                    # 每题 rollout 组大小
BETA_KL = 0.02
LR = 5e-6
MAX_NEW = 160
GROUP_BATCH = 4          # 每步处理多少道题(× G 条轨迹)  # GPU-TUNE
EPOCHS = 1
TEMP = 0.9               # rollout 采样温度(探索)

from ca_agraphrag.kg_tools import KGTools
from ca_agraphrag.env import KGQAEnv
from ca_agraphrag.llm_agent import LLMPolicy, parse_action, ROLE_MAP
from ca_agraphrag.gen_sft_traj import SYS


class HFPolicyModel:
    """包 HF 模型:generate(messages)->(text, gen_ids);sequence_logprobs 供训练。"""
    def __init__(self, model, tok, temperature=TEMP):
        self.model, self.tok, self.temperature = model, tok, temperature

    def generate(self, messages):
        import torch
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=MAX_NEW, do_sample=True,
                                      temperature=self.temperature, top_p=0.95,
                                      pad_token_id=self.tok.eos_token_id)
        gen_ids = out[0, ids.shape[1]:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        self._last_gen_ids = gen_ids
        return text


def rollout_group(item, env, hfmodel, tok, g=G):
    """对一题采样 g 条轨迹。返回 [{token_ids, gen_mask, reward}]。
    token_ids: 整条对话 token;gen_mask: 该 token 是否策略生成(assistant)且计入 loss。"""
    import torch
    samples = []
    for _ in range(g):
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": item["question"]}]
        env.reset(item)
        gen_spans = []          # (start,end) of each generated assistant segment in full ids
        done = False
        for _step in range(env.max_steps):
            prev_ids = tok.apply_chat_template(
                [{"role": ROLE_MAP.get(m["role"], m["role"]), "content": m["content"]} for m in msgs],
                add_generation_prompt=True, return_tensors="pt")[0]
            text = hfmodel.generate([{"role": ROLE_MAP.get(m["role"], m["role"]), "content": m["content"]}
                                     for m in msgs])
            gen_ids = hfmodel._last_gen_ids.cpu()
            start = len(prev_ids)
            gen_spans.append((start, start + len(gen_ids)))
            msgs.append({"role": "assistant", "content": text})
            action = parse_action(text) or {"tool": "finish", "args": {"answer": None}}
            obs, reward, done, info = env.step(action)
            if action["tool"] == "finish" or done:
                break
            msgs.append({"role": "tool", "content": json.dumps({"result": obs.get("result")}, ensure_ascii=False)})
        # 整条 token 序列 + 生成掩码
        full_ids = tok.apply_chat_template(
            [{"role": ROLE_MAP.get(m["role"], m["role"]), "content": m["content"]} for m in msgs],
            add_generation_prompt=False, return_tensors="pt")[0]
        gen_mask = torch.zeros(len(full_ids), dtype=torch.bool)
        for s, e in gen_spans:
            gen_mask[s:min(e, len(full_ids))] = True
        samples.append({"token_ids": full_ids, "gen_mask": gen_mask,
                        "reward": info.get("correctness", 0.0) - 0.02 * (env.steps - 1)})
    return samples


def seq_logprobs(model, ids, gen_mask):
    """整条序列在生成位置的 token logπ。返回 sum 与 count(供归一)。"""
    import torch
    ids = ids.unsqueeze(0).to(model.device)
    logits = model(ids).logits[0, :-1]                    # 预测下一 token
    logp = torch.log_softmax(logits.float(), dim=-1)
    tgt = ids[0, 1:]
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    m = gen_mask[1:].to(tok_logp.device)
    return (tok_logp * m).sum(), m.sum().clamp(min=1)


def grpo_step(policy, ref, samples, optimizer):
    """一组(可含多题×G)样本的 GRPO 更新。samples 已按题分组算优势。"""
    import torch
    loss_terms = []
    for grp in samples:                                    # grp = 一题的 g 条
        rs = torch.tensor([s["reward"] for s in grp])
        adv = (rs - rs.mean()) / (rs.std() + 1e-6)
        for s, a in zip(grp, adv):
            sp, cnt = seq_logprobs(policy, s["token_ids"], s["gen_mask"])
            with torch.no_grad():
                rp, _ = seq_logprobs(ref, s["token_ids"], s["gen_mask"])
            kl = (sp - rp) / cnt                           # 近似 KL(生成位置)
            loss_terms.append(-(a.to(sp.device) * sp / cnt) + BETA_KL * kl)
    loss = torch.stack(loss_terms).mean()
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)   # GPU-TUNE
    optimizer.step()
    return loss.item()


def train():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    policy = PeftModel.from_pretrained(base, str(SFT_ADAPTER), is_trainable=True)   # 可训 LoRA
    ref = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    hfmodel = HFPolicyModel(policy, tok)
    env = KGQAEnv(KGTools(filtered=True))
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=LR)

    items = [json.loads(l) for l in (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    import random
    random.Random(0).shuffle(items)
    step = 0
    for ep in range(EPOCHS):
        buf = []
        for it in items:
            buf.append(rollout_group(it, env, hfmodel, tok))
            if len(buf) >= GROUP_BATCH:
                loss = grpo_step(policy, ref, buf, optimizer)
                step += 1
                avg_r = sum(s["reward"] for g in buf for s in g) / (len(buf) * G)
                print(f"[GRPO] ep{ep} step{step} loss={loss:.4f} avg_reward={avg_r:.3f}", flush=True)
                buf = []
                if step % 100 == 0:
                    policy.save_pretrained(OUT / f"step{step}")
    policy.save_pretrained(OUT / "final")
    print(f"[GRPO] 完成 -> {OUT/'final'}")


if __name__ == "__main__":
    if "--help" in sys.argv:
        print(__doc__)
    else:
        train()
