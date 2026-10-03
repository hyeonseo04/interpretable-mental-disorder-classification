# -*- coding: utf-8 -*-
"""합성 원본 데이터 생성 (파이프라인 테스트 전용 — 실제 데이터 아님).
원본 라벨링 JSON 형식을 흉내 낸다: class, paragraph[index, paragraph_speaker, paragraph_text, <factor>...]"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from imdc.constants import SYMPTOM28  # noqa

KEYF = {"DEPRESSION": SYMPTOM28[:10], "ANXIETY": SYMPTOM28[10:18], "ADDICTION": SYMPTOM28[18:],
        "NORMAL": []}


def main(root, seed=0):
    rnd = random.Random(seed)
    plan = [("D", "DEPRESSION", 60), ("A", "ANXIETY", 56), ("C", "ADDICTION", 59), ("N", "NORMAL", 32)]
    n_sess_target, made = 1440, 0
    persons = [(f"{pre}{i:03d}", cls) for pre, cls, n in plan for i in range(1, n + 1)]
    per = [7] * len(persons)
    for k in range(sum(per) - n_sess_target):     # 1449 → 1440
        per[k] -= 1
    corrupted_left = 15
    for (pid, cls), ns in zip(persons, per):
        for s in range(1, ns + 1):
            split = "Training" if (hash(pid) % 5) else "Validation"
            d = os.path.join(root, split, "02.라벨링데이터", cls)
            os.makedirs(d, exist_ok=True)
            # 장애 내담자 일부 세션은 NORMAL 라벨 (실데이터의 세션 라벨 혼재 흉내)
            scls = "NORMAL" if (cls != "NORMAL" and s == 1 and rnd.random() < 0.3) else cls
            paras, idx = [], 0
            n_utt = rnd.randint(40, 200)
            for u in range(n_utt):
                sp = "내담자" if u % 2 else "상담자"
                p = {"index": (str(idx) if pid.endswith("7") else idx), "paragraph_speaker": sp,
                     "paragraph_text": f"{sp} 발화 {u} " + "가나다라마바사" * rnd.randint(1, 30)}
                if sp == "내담자":
                    for f in SYMPTOM28:
                        base = 0.018 if f in KEYF[cls] else (0.012 if cls != "NORMAL" else 0.008)
                        p[f] = rnd.choice([1, 2, 3]) if rnd.random() < base else 0
                        if p[f]:
                            p["paragraph_text"] += f" <{f}={p[f]}>"
                paras.append(p); idx += 1
            name = f"S{s:02d}_{'raw' if rnd.random() < 0.1 else 'check'}_{pid.lower() if rnd.random() < 0.1 else pid}"
            fp = os.path.join(d, name + ".json")
            if corrupted_left and rnd.random() < 0.02:
                open(fp, "w").write('{"class": "' + scls + '", "paragraph": [')   # 손상 JSON
                corrupted_left -= 1
                # 손상 파일 외에 같은 세션의 정상본은 만들지 않음 → 세션 수가 줄어듦
                continue
            json.dump({"class": scls, "paragraph": paras}, open(fp, "w", encoding="utf-8"), ensure_ascii=False)
            made += 1
    print(f"합성 세션 {made}개 · 손상 {15 - corrupted_left}개 → {root}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synth")
