#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_direct4.py — Direct-to-LR 베이스라인용 4범주 강도 추출.

목적
  Factor(28차원 증상요인) 방식과 "동일한 지도학습 파이프라인"에 투입할
  4차원(우울/불안/중독/정상) 범주 수준 강도 벡터를 LLM으로 추출한다.
  Zero/Few-shot(비지도)과 달리 출력을 다시 LR에 학습시키므로,
  Factor와의 차이가 오직 '분해 granularity(28 vs 4)'로 귀속되는 통제 실험이 된다.

설계 원칙 (공정 비교의 핵심)
  - session_id 정규화·손상 JSON 제외·증분 추출을 vllm_extract.py와 **완전히 동일**하게 유지.
    → 두 방식(Factor / Direct4)이 정확히 같은 세션 집합에서 평가됨을 보장.
  - 출력 스키마도 vllm_extract.py와 동일. 단, "llm" 필드가 28키가 아니라 4키.
      {"session_id","person_id","class","split","llm":{depression,anxiety,addiction,normal}}
    → classify_from_scores.py(또는 열 목록을 인자화한 기존 classify.py)를 그대로 재사용 가능.

사용
CUDA_VISIBLE_DEVICES=0,1 uv run extract_direct4.py \
    ../../data/Training/02.라벨링데이터 \
    ../../data/Validation/02.라벨링데이터 \
    --out session_direct4_14b.jsonl \
    --model Qwen/Qwen2.5-14B-Instruct \
    --tp 2

점검
  jq -r '.session_id' session_direct4_14b.jsonl | sort | uniq -d   # 중복 확인(없어야 정상)
  cat session_direct4_14b.corrupted.txt                           # 손상 제외 목록
"""

import argparse
import glob
import json
import os
import re

# ─────────────────────────────────────────────────────────────
# 상수 정의
# ─────────────────────────────────────────────────────────────
CLIENT_SPEAKER = "내담자"
CLASSES = {"DEPRESSION", "ANXIETY", "ADDICTION", "NORMAL"}

# 4개 범주 수준 축 (28개 증상요인 대신 이 4차원을 추출)
CATEGORIES4 = ["depression", "anxiety", "addiction", "normal"]

CATEGORY_DESC = {
    "depression": "우울 경향 — 지속적 슬픔·무기력·흥미상실 등 우울 관련 신호의 전반적 강도",
    "anxiety": "불안 경향 — 과도한 걱정·초조·긴장·회피 등 불안 관련 신호의 전반적 강도",
    "addiction": "중독 경향 — 조절 실패·갈망·집착 등 중독 관련 신호의 전반적 강도",
    "normal": "정상 상태 — 임상적 문제 없이 안정적으로 기능하는 상태의 전반적 강도",
}

SCALE = ("0 = 나타나지 않음\n1 = 약하게 나타남\n"
         "2 = 어느 정도 뚜렷이 나타남\n3 = 강하게/심하게 나타남")

SYSTEM = ("당신은 한국어 심리상담 대화를 분석하는 임상 전문가입니다. "
          "한 상담 세션에서 내담자가 보인 우울·불안·중독 경향과 정상 상태가 "
          "'얼마나 강하게 나타나는지'를 각각 0~3으로 평가합니다. "
          "누구나 겪는 일시적·일반적 표현은 낮게(0~1), "
          "임상적으로 뚜렷하고 반복적으로 드러나는 경향만 높게(2~3) 평가하세요.")


# ─────────────────────────────────────────────────────────────
# session_id 정규화 — vllm_extract.py와 동일 규약
# ─────────────────────────────────────────────────────────────
def normalize_sid(name: str) -> str:
    sid = name.strip().replace("_raw_", "_check_")
    return re.sub(r"(_check_)([a-z])", lambda m: m.group(1) + m.group(2).upper(), sid)


def person_from_sid(sid: str):
    m = re.search(r"_check_([A-Za-z]\d+)$", sid)
    return m.group(1).upper() if m else None


# ─────────────────────────────────────────────────────────────
# 프롬프트 / 파싱
# ─────────────────────────────────────────────────────────────
def build_messages(transcript: str):
    block = "\n".join(f"- {c}: {CATEGORY_DESC[c]}" for c in CATEGORIES4)
    user = (f"[내담자 발화 모음]\n{transcript}\n\n[평가 척도]\n{SCALE}\n\n"
            f"[평가 항목 4개]\n{block}\n\n"
            "위 세션에 대해 4개 항목의 점수를 매기세요. 설명 없이 JSON 객체 하나만 출력하세요.\n"
            '형식 예: {"depression": 2, "anxiety": 1, "addiction": 0, "normal": 1}')
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


def parse_scores(text: str):
    """LLM 출력에서 4개 범주 점수(0~3 정수)를 파싱.

    Returns
      out    : {category: 0~3}  (실패한 키는 0으로 채움)
      status : "ok"          — JSON 파싱 성공 + 4키 모두 정상 파싱
               "partial"     — JSON은 읽혔으나 일부 키 결측/비정수 (0으로 대체됨)
               "no_json"     — 출력에서 JSON 객체를 찾지 못함 (전부 0)
      n_missing : 0으로 대체된 키 개수
    """
    obj = None
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group())
        except Exception:
            obj = None
    if not isinstance(obj, dict):
        # JSON 자체를 못 읽음 → 전부 0
        return {c: 0 for c in CATEGORIES4}, "no_json", len(CATEGORIES4)

    out, n_missing = {}, 0
    for c in CATEGORIES4:
        if c not in obj:
            out[c] = 0
            n_missing += 1
            continue
        try:
            out[c] = max(0, min(3, int(obj[c])))
        except Exception:
            out[c] = 0                      # 값이 비정수/범위밖 → 0 대체
            n_missing += 1
    status = "ok" if n_missing == 0 else "partial"
    return out, status, n_missing


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
# 세션 수집 (손상 JSON은 자동복구하지 않고 명시적 제외) — vllm_extract.py와 동일
# ─────────────────────────────────────────────────────────────
def collect_jobs(splits, max_chars):
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
    ap = argparse.ArgumentParser(description="4-category direct extraction with vLLM (incremental).")
    ap.add_argument("train_dir", help="Training 라벨링 데이터 디렉토리")
    ap.add_argument("test_dir", nargs="?", default=None, help="Validation 라벨링 데이터 디렉토리(선택)")
    ap.add_argument("--out", default="session_direct4.jsonl", help="출력 JSONL 경로")
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
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len)
    sp = SamplingParams(temperature=0.0, max_tokens=128)

    messages = [build_messages(tr) for *_, tr in todo]
    outs = llm.chat(messages, sp)

    # 4) append 저장 (overwrite면 새로 씀) + 파싱 상태 집계
    mode = "w" if args.overwrite else "a"
    n = 0
    n_ok = n_partial = n_nojson = 0
    failed_sids = []          # partial/no_json 세션 (session_id, status, n_missing)
    with open(args.out, mode, encoding="utf-8") as f:
        for (sid, pid, cls, split, _), o in zip(todo, outs):
            scores, status, n_missing = parse_scores(o.outputs[0].text)
            if status == "ok":
                n_ok += 1
            elif status == "partial":
                n_partial += 1
                failed_sids.append((sid, status, n_missing))
            else:  # no_json
                n_nojson += 1
                failed_sids.append((sid, status, n_missing))
            f.write(json.dumps({
                "session_id": sid, "person_id": pid, "class": cls,
                "split": split, "llm": scores,
                "parse_status": status,          # ok / partial / no_json (다운스트림 필터용)
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"[저장] {n}개 추가 → {args.out}")

    # 파싱 실패 로그
    n_fail = n_partial + n_nojson
    rate = (n_fail / n * 100) if n else 0.0
    print(f"[파싱] 성공 {n_ok} · 부분실패 {n_partial} · JSON없음 {n_nojson}"
          f"  → 실패율 {n_fail}/{n} ({rate:.1f}%)")
    if failed_sids:
        fail_path = f"{base}.parsefail.txt"
        with open(fail_path, "w", encoding="utf-8") as f:
            f.write("# session_id\tstatus\tn_missing_keys\n")
            for sid, status, nm in sorted(failed_sids):
                f.write(f"{sid}\t{status}\t{nm}\n")
        print(f"       파싱 실패 세션 목록 → {fail_path}")
        if rate >= 5.0:
            print(f"       ⚠ 실패율이 5%를 넘습니다. Direct4가 부당하게 불리해질 수 있으니 "
                  f"프롬프트/‧max_tokens 점검을 권장합니다.")

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