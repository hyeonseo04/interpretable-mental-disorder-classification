# -*- coding: utf-8 -*-
"""테스트용 가짜 vLLM — 전사에 심어둔 <factor=v> 표식을 읽어 잡음 섞인 출력을 만든다.
MOCK_CRASH_AFTER=N 이면 N번째 chat 호출 후 예외 (재개 테스트)."""
import hashlib
import json
import os
import random
import re

_calls = 0


class SamplingParams:
    def __init__(self, temperature=1.0, top_p=1.0, max_tokens=16, seed=None, **kw):
        self.temperature, self.top_p, self.max_tokens, self.seed = temperature, top_p, max_tokens, seed


class _Tok:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "".join(f"<|{m['role']}|>{m['content']}" for m in msgs) + "<|assistant|>"

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text) // 2))


class _O:
    def __init__(self, text, fr="stop"):
        self.text, self.finish_reason, self.token_ids = text, fr, list(range(len(text) // 3))


class _R:
    def __init__(self, text, n_prompt):
        self.outputs, self.prompt_token_ids = [_O(text)], list(range(n_prompt))


SYM = ["depressive_mood", "worthlessness", "guilt", "impaired_cognition", "suicidal", "anhedonia",
       "psychomotor_changes", "weight_appetite", "sleep_disturbance", "fatigue",
       "anxiety_mood", "derealization", "perceived_loss_of_control", "anxiety_control",
       "concentration", "avoidance", "physical_symptoms", "irritability",
       "loss_of_control", "craving", "lying", "tolerance", "withdrawal", "salience",
       "resource_investment", "daily_functioning", "social_problems", "negative_consequences"]


class LLM:
    def __init__(self, model, **kw):
        self.model, self.kw = model, kw
        self.noise = 0.5 if "32" in model else 0.3

    def get_tokenizer(self):
        return _Tok()

    def chat(self, batch, sp, use_tqdm=True):
        global _calls
        _calls += 1
        lim = int(os.environ.get("MOCK_CRASH_AFTER", "0"))
        out = []
        for msgs in batch:
            user = msgs[-1]["content"]
            tr = user.split("[내담자 발화 모음]\n", 1)[1]
            rnd = random.Random(hashlib.md5((tr + self.model).encode()).hexdigest())
            marks = {}
            for f, v in re.findall(r"<(\w+)=(\d)>", tr):
                marks[f] = max(marks.get(f, 0), int(v))
            dom = max(("DEPRESSION", sum(marks.get(f, 0) for f in SYM[:10])),
                      ("ANXIETY", sum(marks.get(f, 0) for f in SYM[10:18])),
                      ("ADDICTION", sum(marks.get(f, 0) for f in SYM[18:])), key=lambda x: x[1])
            if "28개" in user:
                d = {f: max(0, min(3, marks.get(f, 0) - (1 if rnd.random() < self.noise else 0)
                                   + (1 if rnd.random() < (0.02 if "32" in self.model else 0.06) else 0))) for f in SYM}
                text = json.dumps(d) if rnd.random() > 0.01 else "죄송합니다"
            elif "4개 항목" in user:
                text = json.dumps({"depression": 2 if dom[0] == "DEPRESSION" and dom[1] else 0,
                                   "anxiety": 2 if dom[0] == "ANXIETY" and dom[1] else 0,
                                   "addiction": 2 if dom[0] == "ADDICTION" and dom[1] else 0,
                                   "normal": 0 if dom[1] else 2})
            else:
                c = dom[0] if dom[1] and rnd.random() > self.noise else rnd.choice(
                    ["DEPRESSION", "ANXIETY", "ADDICTION", "NORMAL"])
                text = json.dumps({"class": c})
            n_prompt = sum(len(m["content"]) for m in msgs) // 2
            out.append(_R(text, n_prompt))
        if lim and _calls >= lim:
            raise RuntimeError("MOCK crash")
        return out
