#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classify_llm_direct_fewshot.py — Few-shot LLM 직접 분류 baseline (고정 합성 예시)

  · 예시는 실제 세션이 아닌 합성 발화(EXAMPLES) → 데이터 누수 없음, 논문 게재 자유
  · 4-shot(범주별 1개), turn 기반 프롬프트
  · vllm_extract.py / zero-shot 과 동일한 데이터 처리 규약:
      - session_id 파일명 기준 + 정규화(_raw_→_check_, person 대문자)
      - 손상 JSON 명시적 제외(*.corrupted.txt)
      - person_id 대문자 통일
      - 출력에 session_id 포함
  · 파싱 실패는 NORMAL 처리하되, 실패 세션의 실제 클래스 분포를 출력(NORMAL 부풀림 점검)

사용:
  CUDA_VISIBLE_DEVICES=0 uv run classify_llm_direct_fewshot.py \
      ../data/Training/02.라벨링데이터 ../data/Validation/02.라벨링데이터 \
      --out result/session_direct_fewshot_14b.jsonl \
      --model Qwen/Qwen2.5-14B-Instruct --tp 1 --gpu-mem 0.72
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict, Counter

from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

CLIENT_SPEAKER = "내담자"
CLASSES = ["DEPRESSION", "ANXIETY", "ADDICTION", "NORMAL"]
CLASS_SET = set(CLASSES)

SYSTEM = ("당신은 한국어 심리상담 대화를 분석하는 임상 전문가입니다. "
          "내담자의 발화를 보고 가장 적합한 범주 하나로 분류하세요.")

# ── 실제 내담자 발화가 아닌 합성 예시(누수 없음) ──
EXAMPLES = {
    "DEPRESSION": [
        "요즘은 뭘 해도 재미가 없고 예전에 좋아하던 것들도 다 시들해졌어요.",
        "아침에 일어나는 것조차 버겁고 하루 종일 무기력해서 아무것도 손에 안 잡혀요.",
        "제가 쓸모없는 사람 같고, 뭘 해도 안 될 것 같다는 생각이 자꾸 들어요.",
    ],
    "ANXIETY": [
        "별일 아닌데도 자꾸 안 좋은 일이 생길 것 같아서 마음이 조마조마해요.",
        "긴장하면 심장이 두근거리고 손이 떨리고 숨이 잘 안 쉬어질 때가 있어요.",
        "걱정을 멈추려고 해도 머릿속에서 계속 맴돌아서 잠도 잘 못 자요.",
    ],
    "ADDICTION": [
        "눈 뜨자마자 핸드폰부터 켜고, 안 하면 불안하고 자꾸 생각나요.",
        "그만해야지 하면서도 멈출 수가 없어서 새벽까지 하게 돼요.",
        "그것 때문에 할 일도 자꾸 미루고 일상에 지장이 생기는데도 계속하게 돼요.",
    ],
    "NORMAL": [
        "요즘 일이 좀 바빴는데 주말에 푹 쉬니까 다시 괜찮아졌어요.",
        "고민이 있긴 한데 친구들이랑 얘기하다 보면 풀리는 편이에요.",
        "예전에 힘든 일도 있었지만 지금 돌아보면 그러면서 좀 성장한 것 같아요.",
    ],
}


def normalize_sid(name: str) -> str:
    sid = name.strip().replace("_raw_", "_check_")
    return re.sub(r"(_check_)([a-z])", lambda m: m.group(1) + m.group(2).upper(), sid)


def person_from_sid(sid: str):
    m = re.search(r"_check_([A-Za-z]\d+)$", sid)
    return m.group(1).upper() if m else None


def load_client_transcript(record: dict, max_chars: int) -> str:
    paras = sorted(record.get("paragraph", []), key=lambda p: p.get("index", 0))
    utts = []
    for p in paras:
        if p.get("paragraph_speaker") != CLIENT_SPEAKER:
            continue
        t = str(p.get("paragraph_text") or "").strip()
        if t:
            utts.append(t)
    return "\n".join(utts)[:max_chars]


def user_prompt(transcript: str):
    return (f"[내담자 발화 모음]\n{transcript}\n\n"
            "이 내담자를 다음 네 범주 중 하나로 분류하세요:\n"
            "- DEPRESSION (우울)\n- ANXIETY (불안)\n- ADDICTION (중독)\n- NORMAL (정상)\n\n"
            '설명 없이 JSON 객체 하나만 출력하세요. 예: {"class": "DEPRESSION"}')


def build_fewshot(transcript: str):
    msgs = [{"role": "system", "content": SYSTEM}]
    for cls in CLASSES:
        ex_tr = "\n".join(EXAMPLES[cls])
        msgs.append({"role": "user", "content": user_prompt(ex_tr)})
        msgs.append({"role": "assistant", "content": json.dumps({"class": cls}, ensure_ascii=False)})
    msgs.append({"role": "user", "content": user_prompt(transcript)})
    return msgs


def parse_class(text: str):
    obj = {}
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group())
        except Exception:
            obj = {}
    c = str(obj.get("class", "") if isinstance(obj, dict) else "").upper()
    if c in CLASS_SET:
        return c
    for cand in CLASSES:
        if cand in text.upper():
            return cand
    return None


def collect_jobs(splits, max_chars):
    jobs, corrupted, empty, seen = [], [], [], set()
    for _, d in splits:
        for fp in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            sid = normalize_sid(os.path.splitext(os.path.basename(fp))[0])
            try:
                record = json.load(open(fp, encoding="utf-8"))
            except Exception:
                corrupted.append(sid)
                continue
            if record.get("class") not in CLASS_SET:
                continue
            if sid in seen:
                continue
            seen.add(sid)
            tr = load_client_transcript(record, max_chars)
            if not tr:
                empty.append(sid)
                continue
            jobs.append((sid, person_from_sid(sid), record["class"], tr))
    return jobs, corrupted, empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_dir")
    ap.add_argument("test_dir", nargs="?", default=None)
    ap.add_argument("--out", default="result/session_direct_fewshot.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.72)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-chars", type=int, default=8000)
    args = ap.parse_args()

    splits = [("train", args.train_dir)] + ([("test", args.test_dir)] if args.test_dir else [])

    jobs, corrupted, empty = collect_jobs(splits, args.max_chars)
    print(f"[수집] 평가 세션 {len(jobs)} · 손상 제외 {len(corrupted)} · 발화없음 제외 {len(empty)} "
          "(합성 예시 사용 — 예시로 인한 제외 없음)")
    base = os.path.splitext(args.out)[0]
    if corrupted:
        open(f"{base}.corrupted.txt", "w", encoding="utf-8").write("\n".join(sorted(corrupted)))

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len, dtype="bfloat16")
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    outs = llm.chat([build_fewshot(tr) for *_, tr in jobs], sp)

    person_pred, person_true = defaultdict(list), {}
    n_fail = 0
    fail_by_true = Counter()
    with open(args.out, "w", encoding="utf-8") as f:
        for (sid, pid, true_c, _), o in zip(jobs, outs):
            raw = o.outputs[0].text
            parsed = parse_class(raw)
            pred = parsed or "NORMAL"
            if parsed is None:
                n_fail += 1
                fail_by_true[true_c] += 1
            person_pred[pid].append(pred); person_true[pid] = true_c
            f.write(json.dumps({
                "session_id": sid, "person_id": pid, "true": true_c,
                "pred": pred, "parsed_ok": parsed is not None, "raw": raw,
            }, ensure_ascii=False) + "\n")

    print(f"\n파싱 실패 {n_fail}/{len(jobs)}건 → NORMAL 처리")
    if n_fail:
        print("  실패 세션 실제 클래스 분포:",
              ", ".join(f"{c} {fail_by_true[c]}" for c in CLASSES if fail_by_true[c]))

    pids = list(person_pred)
    pp = [Counter(person_pred[p]).most_common(1)[0][0] for p in pids]
    tt = [person_true[p] for p in pids]
    print(f"\n[사람] macro-F1 {f1_score(tt, pp, average='macro'):.3f}"
          f" · acc {accuracy_score(tt, pp):.3f} (n={len(pids)}명, 다수결)")

    P, R, F, S = precision_recall_fscore_support(tt, pp, labels=CLASSES, zero_division=0)
    print(f"\n{'class':12}{'P':>8}{'R':>8}{'F1':>8}{'n':>6}")
    for i, c in enumerate(CLASSES):
        print(f"{c:12}{P[i]:8.3f}{R[i]:8.3f}{F[i]:8.3f}{S[i]:6d}")
    print(f"{'macro':12}{P.mean():8.3f}{R.mean():8.3f}{F.mean():8.3f}")

    cm = confusion_matrix(tt, pp, labels=CLASSES)
    print("\n[사람 혼동행렬] (행=정답, 열=예측)")
    print(" " * 12 + "".join(c[:4].rjust(7) for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(c[:11].ljust(12) + "".join(str(cm[i, j]).rjust(7) for j in range(len(CLASSES))))


if __name__ == "__main__":
    main()