#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vllm_extract.py — 상담 세션에서 28개 증상요인(0~3 강도)을 LLM으로 추출.

주요 특징
  - session_id는 **파일명 기준**으로 생성하고 정규화한다.
      * '_raw_' → '_check_' 통일
      * person_id 부분 대문자화 (x001 → X001)
    → 데이터셋에 섞인 표기 불일치(raw/check, 대소문자)로 인한 중복·누락 방지.
  - JSON이 손상되어 표준 파서로 읽히지 않는 세션은 **자동 복구하지 않고 명시적으로 제외**하고,
    제외 목록을 파일로 남긴다(재현성·리뷰 대응).
  - Train/Validation 두 디렉토리를 한 번에 처리하고 각 세션에 split('train'/'test')을 표기한다.
  - **증분 추출**: 출력 파일에 이미 존재하는 session_id는 건너뛰고, 없는 세션만 추출해 append 한다.
    (--overwrite 로 전체 재추출 가능)

사용
CUDA_VISIBLE_DEVICES=0,1 uv run vllm_extract.py \
    ../../data/Training/02.라벨링데이터 \
    ../../data/Validation/02.라벨링데이터 \
    --out session_llm_14b.jsonl \
    --model Qwen/Qwen2.5-14B-Instruct \
    --tp 2

점검
  jq -r '.session_id' session_llm_14b.jsonl | sort | uniq -d      # 중복 확인(없어야 정상)
  cat session_llm_14b.corrupted.txt                              # 손상되어 제외된 세션 목록
"""

import argparse
import glob
import json
import os
import re
import sys

# ─────────────────────────────────────────────────────────────
# 상수 정의
# ─────────────────────────────────────────────────────────────
CLIENT_SPEAKER = "내담자"
CLASSES = {"DEPRESSION", "ANXIETY", "ADDICTION", "NORMAL"}

SYMPTOM28 = [
    "depressive_mood", "worthlessness", "guilt", "impaired_cognition", "suicidal", "anhedonia",
    "psychomotor_changes", "weight_appetite", "sleep_disturbance", "fatigue",
    "anxiety_mood", "derealization", "perceived_loss_of_control", "anxiety_control",
    "concentration", "avoidance", "physical_symptoms", "irritability",
    "loss_of_control", "craving", "lying", "tolerance", "withdrawal", "salience",
    "resource_investment", "daily_functioning", "social_problems", "negative_consequences",
]

FACTOR_DESC = {
    "depressive_mood": "우울한 기분 — 지속적인 슬픔·공허·가라앉음·절망감",
    "worthlessness": "무가치감 — 자신이 쓸모없거나 가치 없다는 느낌",
    "guilt": "죄책감 — 과도한 자책이나 미안함",
    "impaired_cognition": "사고력저하 — 생각이 느려지거나 판단·결정이 평소보다 어려움",
    "suicidal": "자살생각 — 죽고 싶다는 생각이나 자해 충동",
    "anhedonia": "흥미감소 — 평소 즐기던 일에 대한 흥미·즐거움 상실",
    "psychomotor_changes": "정신운동변화 — 말·움직임이 눈에 띄게 느리거나 반대로 안절부절못함",
    "weight_appetite": "체중/식욕변화 — 식욕 저하나 과식, 체중 변화",
    "sleep_disturbance": "수면문제 — 불면이나 과다수면",
    "fatigue": "피로감 — 지속적인 기력 저하, 쉽게 지침",
    "anxiety_mood": "불안감 — 막연한 걱정을 넘어 지속적·과도한 불안·초조·긴장",
    "derealization": "비현실감 — 주변이 비현실적이거나 자신과 분리된 듯한 느낌",
    "perceived_loss_of_control": "통제력상실감 — 상황·자신을 통제 못 한다는 두려움, 곧 큰일 날 것 같음",
    "anxiety_control": "불안조절곤란 — 불안·걱정을 스스로 멈추거나 다스리기 어려움",
    "concentration": "집중력저하 — 불안으로 주의가 흩어지고 집중이 어려움",
    "avoidance": "사회적상황회피 — 불안 때문에 특정 상황·장소·사람을 피함",
    "physical_symptoms": "신체증상 — 불안으로 인한 심계항진·떨림·발한·호흡곤란",
    "irritability": "과민성 — 사소한 일에 쉽게 짜증·예민·화",
    "loss_of_control": "조절실패 — 특정 행동을 줄이거나 멈추려 해도 못 함",
    "craving": "갈망 — 하고 싶은 강한 충동",
    "lying": "거짓말 — 행동을 숨기거나 속임",
    "tolerance": "내성 — 같은 효과에 점점 더 많이 필요로 함",
    "withdrawal": "금단 — 중단 시 불편·신체·정서 증상",
    "salience": "현저성 — 그 행동이 생활의 중심이 됨",
    "resource_investment": "자원투자 — 시간·돈을 과도하게 씀",
    "daily_functioning": "자기관리 저하 — 위생·식사·수면·일상 의무 등 기본 기능 수행 곤란",
    "social_problems": "사회적문제발생 — 대인·직무·가정 문제 발생",
    "negative_consequences": "부정적 결과 — 피해를 알면서도 행동을 지속",
}

SCALE = ("0 = 증상이 나타나지 않음\n1 = 약하게 나타남\n"
         "2 = 어느 정도 뚜렷이 나타남\n3 = 강하게/심하게 나타남")

SYSTEM = ("당신은 한국어 심리상담 대화를 분석하는 임상 전문가입니다. "
          "한 상담 세션에서 내담자가 보인 아래 28개 증상요인이 "
          "'얼마나 강하게 나타나는지(증상의 정도)'를 0~3으로 평가합니다. "
          "누구나 겪는 일시적·일반적 표현은 낮게(0~1), "
          "임상적으로 뚜렷하고 반복적으로 드러나는 증상만 높게(2~3) 평가하세요.")


# ─────────────────────────────────────────────────────────────
# session_id 정규화 — 데이터셋 표기 불일치 흡수
# ─────────────────────────────────────────────────────────────
def normalize_sid(name: str) -> str:
    """파일명(확장자 제외)을 표준 session_id로 변환.
    - '_raw_' → '_check_'
    - person_id 부분 소문자를 대문자로 (x001 → X001)
    """
    sid = name.strip().replace("_raw_", "_check_")
    sid = re.sub(r"(_check_)([a-z])", lambda m: m.group(1) + m.group(2).upper(), sid)
    return sid


def person_from_sid(sid: str):
    """정규화된 session_id 끝에서 person_id를 추출(대문자)."""
    m = re.search(r"_check_([A-Za-z]\d+)$", sid)
    return m.group(1).upper() if m else None


# ─────────────────────────────────────────────────────────────
# 프롬프트 / 파싱
# ─────────────────────────────────────────────────────────────
def build_messages(transcript: str):
    block = "\n".join(f"- {f}: {FACTOR_DESC[f]}" for f in SYMPTOM28)
    user = (f"[내담자 발화 모음]\n{transcript}\n\n[평가 척도]\n{SCALE}\n\n"
            f"[증상요인 28개]\n{block}\n\n"
            "위 세션에서 내담자가 보인 28개 요인의 점수를 매기세요. 설명 없이 JSON 객체 하나만 출력하세요.\n"
            '형식 예: {"depressive_mood": 0, "anxiety_mood": 2, ... (28개 모두 포함)}')
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


def parse_scores(text: str):
    """LLM 출력에서 28개 요인 점수(0~3 정수)를 파싱. 실패 값은 0으로 채움."""
    obj = {}
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group())
        except Exception:
            obj = {}
    out = {}
    for f in SYMPTOM28:
        v = obj.get(f, 0)
        try:
            v = int(v)
        except Exception:
            v = 0
        out[f] = max(0, min(3, v))
    return out


def load_client_transcript(record: dict, max_chars: int) -> str:
    """세션 레코드에서 내담자 발화만 index 순서로 연결."""
    paras = sorted(record.get("paragraph", []), key=lambda p: p.get("index", 0))
    utts = []
    for p in paras:
        if p.get("paragraph_speaker") != CLIENT_SPEAKER:
            continue
        t = str(p.get("paragraph_text") or "").strip()
        if t:
            utts.append(t)
    return "\n".join(utts)[:max_chars]


# ─────────────────────────────────────────────────────────────
# 세션 수집 (손상 JSON은 자동복구하지 않고 명시적 제외)
# ─────────────────────────────────────────────────────────────
def collect_jobs(splits, max_chars):
    """
    Returns
      jobs      : [(session_id, person_id, class, split, transcript), ...]
      corrupted : [session_id, ...]   # JSON 파싱 실패로 제외
      empty     : [session_id, ...]   # 내담자 발화가 없어 제외
      seen_ids  : set                 # 이번 수집에서 확인된 전체 유효 session_id
    """
    jobs, corrupted, empty = [], [], []
    seen = set()
    for split, d in splits:
        files = sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True))
        for fp in files:
            sid = normalize_sid(os.path.splitext(os.path.basename(fp))[0])
            try:
                record = json.load(open(fp, encoding="utf-8"))
            except Exception:
                corrupted.append(sid)          # 손상 → 제외(복구하지 않음)
                continue
            if record.get("class") not in CLASSES:
                continue
            if sid in seen:
                continue                        # 정규화 후 중복 제거
            seen.add(sid)
            transcript = load_client_transcript(record, max_chars)
            if not transcript:
                empty.append(sid)               # 내담자 발화 없음 → 제외
                continue
            pid = person_from_sid(sid)
            jobs.append((sid, pid, record["class"], split, transcript))
    return jobs, corrupted, empty, seen


def load_existing_ids(out_path: str) -> set:
    """출력 파일에 이미 저장된 session_id 집합(증분 추출용)."""
    ids = set()
    if not os.path.exists(out_path):
        return ids
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(normalize_sid(json.loads(line)["session_id"]))
            except Exception:
                continue
    return ids


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="28-factor extraction with vLLM (incremental).")
    ap.add_argument("train_dir", help="Training 라벨링 데이터 디렉토리")
    ap.add_argument("test_dir", nargs="?", default=None, help="Validation 라벨링 데이터 디렉토리(선택)")
    ap.add_argument("--out", default="session_llm.jsonl", help="출력 JSONL 경로")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size (노출 GPU 수와 일치)")
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-chars", type=int, default=8000, help="transcript 최대 길이(문자)")
    ap.add_argument("--overwrite", action="store_true", help="기존 결과 무시하고 전체 재추출")
    args = ap.parse_args()

    splits = [("train", args.train_dir)] + ([("test", args.test_dir)] if args.test_dir else [])

    # 1) 전체 유효 세션 수집
    jobs, corrupted, empty, seen = collect_jobs(splits, args.max_chars)
    print(f"[수집] 유효 세션 {len(jobs)} · 손상 제외 {len(corrupted)} · 발화없음 제외 {len(empty)}")

    # 손상/발화없음 목록 기록(재현성)
    base = os.path.splitext(args.out)[0]
    if corrupted:
        with open(f"{base}.corrupted.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(corrupted)))
        print(f"       손상 세션 목록 → {base}.corrupted.txt")
    if empty:
        with open(f"{base}.empty.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(empty)))
        print(f"       발화없음 목록 → {base}.empty.txt")

    # 2) 증분: 이미 추출된 세션 제외
    existing = set() if args.overwrite else load_existing_ids(args.out)
    todo = [j for j in jobs if j[0] not in existing]
    print(f"[증분] 기존 {len(existing)} · 이번에 추출 {len(todo)}"
          + (" (overwrite)" if args.overwrite else ""))

    if not todo:
        print("추출할 새 세션이 없습니다. 종료.")
        return

    # 3) vLLM 로드 & 추출
    from vllm import LLM, SamplingParams  # 무거운 import는 필요 시점에
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len)
    sp = SamplingParams(temperature=0.0, max_tokens=512)

    messages = [build_messages(tr) for *_, tr in todo]
    outs = llm.chat(messages, sp)

    # 4) append 저장 (overwrite면 새로 씀)
    mode = "w" if args.overwrite else "a"
    n = 0
    with open(args.out, mode, encoding="utf-8") as f:
        for (sid, pid, cls, split, _), o in zip(todo, outs):
            scores = parse_scores(o.outputs[0].text)
            f.write(json.dumps({
                "session_id": sid, "person_id": pid, "class": cls,
                "split": split, "llm": scores,
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"[저장] {n}개 추가 → {args.out}")

    # 5) 최종 정합성 점검
    final_ids = load_existing_ids(args.out)
    print(f"[검증] 출력 파일 고유 세션 {len(final_ids)} / 전체 유효 세션 {len(seen)}")
    missing = seen - final_ids
    if missing:
        print(f"        아직 누락 {len(missing)}개: {sorted(missing)[:10]} ...")
    else:
        print("        누락 없음 ✓")


if __name__ == "__main__":
    main()