# -*- coding: utf-8 -*-
"""
data.py — 원본 라벨링 JSON → 실험 정본(manifest) 생성.  LLM 추론 없음.

모든 추론(8개 구성)과 모든 분석이 이 매니페스트 하나만 읽는다.
→ 8개 구성이 정확히 같은 세션·같은 전사·같은 폴드에서 평가됨이 구조적으로 보장된다.

세션 수집 규약
  - Training → Validation 순, 각 디렉토리 안에서 sorted(glob('**/*.json'))
  - session_id = 파일명 정규화 ('_raw_'→'_check_', person 부분 대문자)
  - JSON 파싱 실패 → 복구하지 않고 제외(corrupted 목록 기록)
  - class ∉ {DEP, ANX, ADD, NOR} → 제외 / 정규화 후 중복 → 첫 번째만 사용
  - 내담자 발화(paragraph_speaker == '내담자')만 index 순으로 '\n' 연결
  - 전사를 자르지 않고 전체를 사용한다

추가 산출물
  - gold  : 세션 골드 = 내담자 발화(텍스트 비어있지 않은 것)별 요인 값의 **최댓값** (0~3)
  - folds : 내담자 기준 GroupKFold(5) 배정 — 모든 LR 실험·분석이 이 파일을 공유
  - person_label : 장애 우선 규칙 (장애 세션이 하나라도 있으면 그 장애, 없으면 NORMAL)

사용
  python -m imdc.data data/Training/02.라벨링데이터 data/Validation/02.라벨링데이터 \
      --out-dir data/processed [--compare-gold results/session_gold.jsonl]
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from .constants import (CLASS_SET, CLIENT_SPEAKER, EXPECTED_N_CORRUPTED, EXPECTED_N_PERSONS,
                        EXPECTED_N_SESSIONS, N_FOLDS, SYMPTOM28)


# ── session_id 규약 ─────────────────────────────────────────────
def normalize_sid(name: str) -> str:
    sid = str(name).strip().replace("_raw_", "_check_")
    return re.sub(r"(_check_)([a-z])", lambda m: m.group(1) + m.group(2).upper(), sid)


def person_from_sid(sid: str):
    m = re.search(r"_check_([A-Za-z]\d+)$", sid)
    return m.group(1).upper() if m else None


def _index_key_numeric(p):
    v = p.get("index", 0)
    try:
        return (0, int(v))
    except (TypeError, ValueError):
        return (1, str(v))


def _index_key_original(p):
    return p.get("index", 0)          # 원시 index 값 정렬 키 (문자열이면 사전식)


def client_utterances(record: dict):
    """[(text, {factor: gold>0}), ...] — 내담자 발화만, index 숫자 순."""
    paras = [p for p in record.get("paragraph", []) if isinstance(p, dict)]
    paras = sorted(paras, key=_index_key_numeric)
    out = []
    for p in paras:
        if p.get("paragraph_speaker") != CLIENT_SPEAKER:
            continue
        t = str(p.get("paragraph_text") or "").strip()
        if not t:
            continue
        gold = {}
        for f in SYMPTOM28:
            v = p.get(f)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and v > 0:
                gold[f] = int(v)
        out.append((t, gold))
    return out, paras


def _order_differs(paras):
    """원시 index 정렬과 숫자 정렬 결과가 다른지 — index 가 문자열로 저장된 경우 탐지."""
    try:
        orig = sorted(paras, key=_index_key_original)
    except TypeError:
        return True
    return [id(p) for p in orig] != [id(p) for p in paras]


def resolve_dir(d):
    """macOS 에서 복사한 한글 폴더명(NFD)과 스크립트 경로(NFC)가 달라도 찾도록."""
    if os.path.isdir(d):
        return d
    for form in ("NFC", "NFD"):
        cand = unicodedata.normalize(form, d)
        if os.path.isdir(cand):
            return cand
    return d


def collect(splits):
    sessions, corrupted, empty, seen = [], [], [], set()
    n_index_str = n_order_diff = 0
    for split, d in splits:
        d = resolve_dir(d)
        files = sorted(f for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True)
                       if not os.path.basename(f).startswith("._"))   # macOS 메타파일 제외
        if not files:
            sys.exit(f"[오류] {d} 에서 *.json 을 찾지 못했습니다.")
        for fp in files:
            sid = normalize_sid(os.path.splitext(os.path.basename(fp))[0])
            try:
                with open(fp, encoding="utf-8") as fh:
                    rec = json.load(fh)
            except Exception:
                corrupted.append(sid)
                continue
            if rec.get("class") not in CLASS_SET:
                continue
            if sid in seen:
                continue
            seen.add(sid)
            utts, paras = client_utterances(rec)
            if any(isinstance(p.get("index"), str) for p in paras):
                n_index_str += 1
            if _order_differs(paras):
                n_order_diff += 1
            if not utts:
                empty.append(sid)
                continue
            transcript = "\n".join(t for t, _ in utts)
            gold = {f: 0 for f in SYMPTOM28}
            for _, g in utts:
                for f, v in g.items():
                    gold[f] = max(gold[f], v)
            sessions.append({
                "session_id": sid, "person_id": person_from_sid(sid), "class": rec["class"],
                "split": split, "n_utts": len(utts), "n_chars": len(transcript),
                "transcript_sha1": hashlib.sha1(transcript.encode("utf-8")).hexdigest(),
                "transcript": transcript, "gold": gold,
            })
    return sessions, corrupted, empty, n_index_str, n_order_diff


def person_labels(sessions):
    """장애 우선 규칙. 서로 다른 장애가 섞인 내담자는 정의가 불가능하므로 오류."""
    pc = defaultdict(set)
    for s in sessions:
        pc[s["person_id"]].add(s["class"])
    labels, mixed_normal = {}, []
    for p, cs in pc.items():
        dis = cs - {"NORMAL"}
        if len(dis) > 1:
            raise ValueError(f"내담자 {p} 에 서로 다른 장애 범주가 섞여 있음: {cs}")
        labels[p] = next(iter(dis)) if dis else "NORMAL"
        if dis and "NORMAL" in cs:
            mixed_normal.append(p)
    return labels, mixed_normal


def make_folds(sessions, k=N_FOLDS):
    """GroupKFold(k) — session_id 정렬 순서로 고정. {person_id: fold}"""
    from sklearn.model_selection import GroupKFold
    ss = sorted(sessions, key=lambda s: s["session_id"])
    groups = np.array([s["person_id"] for s in ss])
    X = np.zeros((len(ss), 1))
    fold_of = {}
    for f, (_, te) in enumerate(GroupKFold(n_splits=k).split(X, groups=groups)):
        for p in set(groups[te]):
            fold_of[p] = f
    return fold_of


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Build experiment manifest (no truncation).")
    ap.add_argument("train_dir")
    ap.add_argument("test_dir", nargs="?", default=None)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--compare-gold", default=None,
                    help="다른 session_gold.jsonl 과 세션 골드 일치 여부 비교")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="세션/내담자 수가 논문과 달라도 계속 (기본: 중단)")
    args = ap.parse_args()

    splits = [("train", args.train_dir)] + ([("test", args.test_dir)] if args.test_dir else [])
    sessions, corrupted, empty, n_index_str, n_order_diff = collect(splits)
    plabel, mixed = person_labels(sessions)

    print(f"[수집] 세션 {len(sessions)} · 내담자 {len(plabel)} · 손상 제외 {len(corrupted)} "
          f"· 발화없음 제외 {len(empty)}")
    print(f"       (논문: 세션 {EXPECTED_N_SESSIONS} · 내담자 {EXPECTED_N_PERSONS} · 손상 {EXPECTED_N_CORRUPTED})")
    ok = (len(sessions), len(plabel), len(corrupted)) == (
        EXPECTED_N_SESSIONS, EXPECTED_N_PERSONS, EXPECTED_N_CORRUPTED)
    if not ok and not args.allow_mismatch:
        sys.exit("[중단] 데이터 규모가 논문과 다릅니다. 경로를 확인하거나 --allow-mismatch 로 진행하세요.")
    print(f"[index] 문자열 index 를 가진 세션 {n_index_str}개 · 원시 index 정렬과 순서가 달라지는 세션 {n_order_diff}개")
    if n_order_diff:
        print("        ⚠ index 가 문자열로 저장된 세션이 있어 숫자 기준으로 정렬했습니다.")
    pcnt = Counter(plabel.values())
    scnt = Counter(s["class"] for s in sessions)
    print("[범주] 내담자 " + ", ".join(f"{c[:3]}={pcnt[c]}" for c in sorted(pcnt))
          + " · 세션 " + ", ".join(f"{c[:3]}={scnt[c]}" for c in sorted(scnt)))
    print(f"[라벨] 장애 내담자 중 NORMAL 세션이 섞인 내담자: {len(mixed)}명")

    folds = make_folds(sessions)
    os.makedirs(args.out_dir, exist_ok=True)
    sessions.sort(key=lambda s: s["session_id"])
    man_path = os.path.join(args.out_dir, "manifest.jsonl")
    with open(man_path, "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps({**s, "person_label": plabel[s["person_id"]],
                                "fold": folds[s["person_id"]]}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "session_gold.jsonl"), "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps({"session_id": s["session_id"], "person_id": s["person_id"],
                                "class": s["class"], "gold": s["gold"]}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "folds.json"), "w", encoding="utf-8") as f:
        json.dump(dict(sorted(folds.items())), f, indent=1)
    meta = {"n_sessions": len(sessions), "n_persons": len(plabel), "n_corrupted": len(corrupted),
            "corrupted": sorted(corrupted), "empty": sorted(empty),
            "persons_per_fold": dict(Counter(folds.values())),
            "n_index_order_changed": n_order_diff, "manifest_sha1": sha1_file(man_path),
            "chars": {"mean": float(np.mean([s["n_chars"] for s in sessions])),
                      "max": int(max(s["n_chars"] for s in sessions))}}
    with open(os.path.join(args.out_dir, "manifest_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[저장] {man_path} (sha1 {meta['manifest_sha1'][:12]}) · session_gold.jsonl · folds.json")
    print(f"       전사 길이(문자) 평균 {meta['chars']['mean']:,.0f} · 최대 {meta['chars']['max']:,} — 절단 없음")

    if args.compare_gold:
        old = {}
        for line in open(args.compare_gold, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                old[normalize_sid(r["session_id"])] = r["gold"]
        new = {s["session_id"]: s["gold"] for s in sessions}
        common = set(old) & set(new)
        diff = [sid for sid in common if any(int(old[sid].get(f, 0)) != new[sid][f] for f in SYMPTOM28)]
        print(f"[골드 비교] 비교 대상 {len(old)} · 신규 {len(new)} · 공통 {len(common)} · 값이 다른 세션 {len(diff)}")
        if diff:
            print("           ⚠ 비교 파일과 골드 값이 다릅니다. 예: " + ", ".join(sorted(diff)[:5]))


if __name__ == "__main__":
    main()
