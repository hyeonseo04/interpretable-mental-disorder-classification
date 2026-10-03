# -*- coding: utf-8 -*-
"""
infer.py — 4개 방법의 vLLM 추론 (전사 절단 없음, 중단 후 재개 가능).

한 번 실행에 모델 하나를 로드하고 --methods 에 적은 방법들을 순서대로 수행한다.
  → 14B 4개 + 32B 4개 = 8개 구성이 모델 로드 2번으로 끝난다 (scripts/run_all.sh).

재개(resume) 설계
  - 세션은 매니페스트의 session_id 정렬 순서로 **고정 크기 청크**(기본 128)로 나뉜다.
  - 청크 하나가 끝날 때마다 결과를 append + fsync 한다. 청크는 all-or-nothing:
    재시작 시 일부만 기록된 청크(중간 크래시)는 버리고 그 청크 전체를 다시 돌린다.
    → 재개해도 각 청크의 배치 구성이 처음 실행과 같다 (greedy 재현성 보호).
  - 출력 옆 {stem}.meta.json 에 모델·스냅샷·버전·샘플링 설정을 기록하고,
    재개 시 설정이 다르면 중단한다 (서로 다른 조건의 결과가 한 파일에 섞이는 것 방지).
  - 전 세션 완료 + 검증 통과 시 {stem}.done 생성. run_all.sh 는 .done 이 있으면 건너뛴다.

사용
  python -m imdc.infer --model-key 14b --model-path /workspace/hf/.../snapshots/<sha> \
      --methods factor28 zeroshot fewshot direct4 --tp 1
  python -m imdc.infer --model-key 14b --model-path ... --methods factor28 --dry-run   # 토큰 길이만 점검
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from collections import Counter

from . import env as envmod
from .constants import (CATEGORIES4, DTYPE, MAX_MODEL_LEN, METHODS, MODELS, SEED, SYMPTOM28,
                        stem as make_stem)
from .prompts import BUILDERS, parse_class, parse_score_vector


# ── I/O ─────────────────────────────────────────────────────────
def load_manifest(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["session_id"])
    return rows


def chunks_of(rows, size):
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def read_valid_records(path):
    """출력 파일에서 온전한 JSON 줄만 읽는다 (마지막 줄이 잘린 경우 대비)."""
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"   ⚠ 손상된 줄 1개 무시 (중단 시점의 잘린 기록)")
    return recs


def rewrite_atomic(path, recs):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


# ── 결과 레코드 ─────────────────────────────────────────────────
def make_record(method, s, out):
    o = out.outputs[0]
    raw = o.text
    base = {"session_id": s["session_id"], "person_id": s["person_id"], "class": s["class"],
            "split": s["split"]}
    meta = {"raw": raw, "finish_reason": getattr(o, "finish_reason", None),
            "n_prompt_tokens": len(out.prompt_token_ids or []),
            "n_gen_tokens": len(getattr(o, "token_ids", []) or []),
            "transcript_sha1": s["transcript_sha1"], "n_chars": s["n_chars"]}
    if method in ("factor28", "direct4"):
        keys = SYMPTOM28 if method == "factor28" else CATEGORIES4
        scores, status, n_missing = parse_score_vector(raw, keys)
        return {**base, "llm": scores, "parse_status": status, "n_missing": n_missing, **meta}
    pred, status = parse_class(raw)
    return {**base, "true": s["class"], "pred": pred or "NORMAL",      # 원본 규칙: 실패 → NORMAL
            "parsed_ok": pred is not None, "parse_status": status, **meta}


# ── 토큰 길이 사전 점검 ─────────────────────────────────────────
def count_prompt_tokens(tok, msgs_list):
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs_list]
    return [len(tok.encode(t, add_special_tokens=False)) for t in texts]


def preflight(tok, method, rows, max_model_len):
    mt = METHODS[method]["max_tokens"]
    lens = count_prompt_tokens(tok, [BUILDERS[method](r["transcript"]) for r in rows])
    worst = max(lens)
    over = [r["session_id"] for r, n in zip(rows, lens) if n + mt > max_model_len]
    print(f"   [토큰] {method}: 입력 평균 {st.mean(lens):,.0f} · 최대 {worst:,} · "
          f"최대+생성 {worst + mt:,} / {max_model_len:,}  → 초과 {len(over)}개")
    if over:
        sys.exit(f"[중단] 컨텍스트 초과 세션 {len(over)}개 (예: {over[:3]}). "
                 "전사를 자르지 않는 것이 원칙이므로 자동 절단하지 않습니다.")
    return {"mean": st.mean(lens), "max": worst, "max_plus_gen": worst + mt}


# ── 실행 설정 일치 검사 ─────────────────────────────────────────
def run_config(args, method, model_path):
    return {"method": method, "model_key": args.model_key, "hf_id": MODELS[args.model_key]["hf_id"],
            "model_path": model_path, "snapshot": envmod.snapshot_of(model_path),
            "max_model_len": args.max_model_len, "max_tokens": METHODS[method]["max_tokens"],
            "temperature": 0.0, "top_p": 1.0, "seed": SEED, "dtype": DTYPE,
            "tp": args.tp, "chunk": args.chunk, "truncation": "none"}


_MUST_MATCH = ("method", "hf_id", "snapshot", "max_model_len", "max_tokens", "temperature",
               "top_p", "seed", "dtype", "tp", "chunk", "truncation")


def check_resume_meta(meta_path, cfg, force):
    if not os.path.exists(meta_path):
        return
    old = json.load(open(meta_path, encoding="utf-8"))["config"]
    diff = {k: (old.get(k), cfg.get(k)) for k in _MUST_MATCH if old.get(k) != cfg.get(k)}
    if diff and not force:
        sys.exit(f"[중단] 기존 부분 결과와 실행 조건이 다릅니다: {diff}\n"
                 "        조건을 맞추거나, 처음부터 다시 돌리려면 출력 파일을 지우세요.")


# ── 메인 루프 ───────────────────────────────────────────────────
def run_method(llm, tok, args, method, rows, model_path):
    stem = make_stem(method, args.model_key, args.suffix)
    out_path = os.path.join(args.out_dir, stem + ".jsonl")
    meta_path = os.path.join(args.out_dir, stem + ".meta.json")
    done_path = os.path.join(args.out_dir, stem + ".done")
    print(f"\n━━ {method} × {args.model_key}  →  {out_path}")
    if os.path.exists(done_path) and not args.redo:
        print("   이미 완료(.done) — 건너뜀")
        return
    cfg = run_config(args, method, model_path)
    check_resume_meta(meta_path, cfg, args.force_resume)

    sha = {r["session_id"]: r["transcript_sha1"] for r in rows}
    chunks = chunks_of(rows, args.chunk)
    chunk_of = {r["session_id"]: i for i, c in enumerate(chunks) for r in c}

    # 기존 결과 정리: 매니페스트와 전사 해시가 맞고, 완결된 청크에 속한 기록만 유지
    existing = read_valid_records(out_path)
    by_sid = {}
    for r in existing:
        sid = r.get("session_id")
        if sid in sha and r.get("transcript_sha1") == sha[sid]:
            by_sid[sid] = r
    complete = {i for i, c in enumerate(chunks) if all(r["session_id"] in by_sid for r in c)}
    keep = [by_sid[r["session_id"]] for i in sorted(complete) for r in chunks[i]]
    if len(keep) != len(existing):
        print(f"   [재개] 기존 {len(existing)}줄 중 {len(keep)}줄 유지 (미완결 청크·불일치 기록 폐기)")
        rewrite_atomic(out_path, keep)
    todo = [i for i in range(len(chunks)) if i not in complete]
    print(f"   [재개] 청크 {len(chunks)}개 중 완료 {len(complete)} · 남음 {len(todo)}")

    if todo:
        tok_stats = preflight(tok, method, [r for i in todo for r in chunks[i]], args.max_model_len)
        info = envmod.collect()
        meta = {"config": cfg, "env": info, "token_stats_todo": tok_stats,
                "manifest_sha1": args.manifest_sha1, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        if os.path.exists(meta_path):
            prev = json.load(open(meta_path, encoding="utf-8"))
            meta["started"] = prev.get("started", meta["started"])
            meta["resumed"] = prev.get("resumed", []) + [time.strftime("%Y-%m-%d %H:%M:%S")]
        json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=METHODS[method]["max_tokens"],
                            seed=SEED)
        build = BUILDERS[method]
        t0 = time.time()
        for k, ci in enumerate(todo, 1):
            batch = chunks[ci]
            outs = llm.chat([build(r["transcript"]) for r in batch], sp, use_tqdm=False)
            assert len(outs) == len(batch)
            with open(out_path, "a", encoding="utf-8") as f:
                for r, o in zip(batch, outs):
                    f.write(json.dumps(make_record(method, r, o), ensure_ascii=False) + "\n")
                f.flush(); os.fsync(f.fileno())
            el = time.time() - t0
            print(f"   청크 {ci + 1}/{len(chunks)} 완료 ({k}/{len(todo)}) · 경과 {el / 60:.1f}분 · "
                  f"남은 예상 {el / k * (len(todo) - k) / 60:.1f}분", flush=True)

    finalize(method, out_path, done_path, meta_path, rows)


def finalize(method, out_path, done_path, meta_path, rows):
    recs = read_valid_records(out_path)
    sids = [r["session_id"] for r in recs]
    dup = [s for s, c in Counter(sids).items() if c > 1]
    missing = set(r["session_id"] for r in rows) - set(sids)
    status = Counter(r.get("parse_status") for r in recs)
    trunc_gen = sum(1 for r in recs if r.get("finish_reason") == "length")
    n = len(recs)
    fail = n - status.get("ok", 0)
    print(f"   [검증] 기록 {n} · 중복 {len(dup)} · 누락 {len(missing)} · 파싱 {dict(status)} "
          f"· 생성 길이 한도 도달 {trunc_gen}")
    summary = {"n": n, "parse_status": dict(status), "parse_fail_rate": fail / max(n, 1),
               "finish_length": trunc_gen,
               "prompt_tokens_max": max((r.get("n_prompt_tokens", 0) for r in recs), default=0),
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    if dup or missing:
        print("   ✗ 완료 표시하지 않음 — 다시 실행하면 남은 청크를 이어서 돌립니다.")
        return
    if fail / max(n, 1) >= 0.05:
        print(f"   ⚠ 파싱 실패율 {fail / n:.1%} ≥ 5% — raw 출력을 확인하세요 (완료 처리는 함).")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        meta["summary"] = summary
        json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(summary, open(done_path, "w", encoding="utf-8"), indent=1)
    print(f"   ✓ 완료 → {done_path}")


def main():
    ap = argparse.ArgumentParser(description="Full-transcript vLLM inference (resumable).")
    ap.add_argument("--manifest", default="data/processed/manifest.jsonl")
    ap.add_argument("--model-key", required=True, choices=sorted(MODELS))
    ap.add_argument("--model-path", default=None,
                    help="로컬 스냅샷 경로 (권장). 없으면 HF id 로 로드")
    ap.add_argument("--methods", nargs="+", default=["factor28", "zeroshot", "fewshot", "direct4"],
                    choices=sorted(METHODS))
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--suffix", default="_full")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    ap.add_argument("--max-num-seqs", type=int, default=None, help="OOM 시 4 등으로 제한")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="앞 N세션만 (스모크 테스트, suffix 에 _smoke 권장)")
    ap.add_argument("--dry-run", action="store_true", help="모델 로드 없이 토큰 길이만 점검")
    ap.add_argument("--redo", action="store_true", help=".done 이 있어도 다시 검증/실행")
    ap.add_argument("--force-resume", action="store_true", help="실행 조건이 달라도 이어서 실행(비권장)")
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    meta_path = os.path.join(os.path.dirname(args.manifest), "manifest_meta.json")
    args.manifest_sha1 = (json.load(open(meta_path))["manifest_sha1"]
                          if os.path.exists(meta_path) else None)
    model_path = args.model_path or MODELS[args.model_key]["hf_id"]
    snap = envmod.snapshot_of(model_path)
    want = MODELS[args.model_key]["revision"]
    print(f"[모델] {model_path}\n       snapshot={snap} (고정 리비전: {want[:12]}…)")
    if snap and snap != want:
        sys.exit(f"[중단] 모델 스냅샷 {snap} 이 고정 리비전 {want} 과 다릅니다. "
                 "scripts/setup_runpod.sh 로 해당 리비전을 받으세요.")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[데이터] 세션 {len(rows)} (전사 절단 없음, 최대 {max(r['n_chars'] for r in rows):,}자)")

    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path)
        for m in args.methods:
            preflight(tok, m, rows, args.max_model_len)
        return

    from vllm import LLM
    kw = dict(model=model_path, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, dtype=DTYPE, seed=SEED)
    if args.max_num_seqs:
        kw["max_num_seqs"] = args.max_num_seqs
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    for m in args.methods:
        run_method(llm, tok, args, m, rows, model_path)


if __name__ == "__main__":
    main()
