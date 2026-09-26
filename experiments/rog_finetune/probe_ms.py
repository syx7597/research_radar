# -*- coding: utf-8 -*-
from modelscope.hub.api import HubApi
api = HubApi()
cands = [
    "modelscope/Llama-2-7b-chat-ms",
    "shakechen/Llama-2-7b-chat-hf",
    "AI-ModelScope/Llama-2-7b-chat-hf",
    "modelscope/Llama-2-7b-ms",
    "shakechen/Llama-2-7b-hf",
    "AI-ModelScope/LLaMA-2-7B-Chat",
]
for mid in cands:
    try:
        fs = api.get_model_files(mid)
        names = [f["Name"] for f in fs]
        st = sum(1 for n in names if n.endswith(".safetensors"))
        bn = sum(1 for n in names if n.endswith(".bin"))
        print(f"OK   {mid}  files={len(names)} safetensors={st} bin={bn}")
    except Exception as e:
        print(f"FAIL {mid}  {type(e).__name__}: {str(e)[:60]}")
