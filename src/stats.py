#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats.py — 분류 성능 통계 (논문 4.1 / 4.2) — 정본(single source of truth)

  ① fold별 macro-F1 mean±std (GroupKFold, session_id 정렬로 폴드 고정)
  ② bootstrap 95% CI (세션·사람 단위: macro-F1, accuracy)
  ③ 계수 그룹-bootstrap + BH-FDR → 클래스별 유의 기여 factor → sig.json 저장
  ④ McNemar: Factor vs zero-shot / few-shot / Direct4(모두 LR 경유 방식은 동일 파이프라인)

정본 규약 (재현성의 핵심)
  - 모든 입력 rows는 session_id 오름차순 정렬 → GroupKFold 폴드 배정 영구 고정
  - Direct4는 zero/few(직접 예측)와 달리 Factor와 동일한 LR 파이프라인으로 학습
  - 사람 단위: LR 계열은 soft voting, 직접 예측 계열은 다수결
  - 논문 수치는 이 스크립트 출력만 사용 (classify_from_scores.py는 폐기 또는 점검용)

사용:
  uv run --with scikit-learn --with scipy stats.py \
      --llm     ../results/session_llm_32b.jsonl \
      --direct4 ../results/session_direct4_32b.jsonl \
      --zero    ../results/session_direct_zeroshot_32b.jsonl \
      --few     ../results/session_direct_fewshot_32b.jsonl \
      --sig-out ../results/sig_32b.json \
      --tag     32b
"""

import argparse
import json
import os
import re
from collections import defaultdict, Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from scipy.stats import chi2, binomtest

SYMPTOM28 = [
    "depressive_mood", "worthlessness", "guilt", "impaired_cognition", "suicidal", "anhedonia",
    "psychomotor_changes", "weight_appetite", "sleep_disturbance", "fatigue",
    "anxiety_mood", "derealization", "perceived_loss_of_control", "anxiety_control",
    "concentration", "avoidance", "physical_symptoms", "irritability",
    "loss_of_control", "craving", "lying", "tolerance", "withdrawal", "salience",
    "resource_investment", "daily_functioning", "social_problems", "negative_consequences",
]
CATEGORIES4 = ["depression", "anxiety", "addiction", "normal"]


# ─────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────
def normalize_sid(name: str) -> str:
    sid = str(name).strip().replace("_raw_", "_check_")
    return re.sub(r"(_check_)([a-z])", lambda m: m.group(1) + m.group(2).upper(), sid)


def load_rows(path, feat_keys):
    """jsonl 로드 + 정규화 + 중복 제거 + session_id 정렬(폴드 고정의 핵심)."""
    rows, seen = [], set()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        sid = normalize_sid(r["session_id"])
        if sid in seen:
            continue
        seen.add(sid)
        r["session_id"] = sid
        r["person_id"] = str(r["person_id"]).upper()
        rows.append(r)
    rows.sort(key=lambda r: r["session_id"])          # ★ 정본 규약: 행 순서 고정
    missing = [k for k in feat_keys if k not in rows[0]["llm"]]
    if missing:
        raise ValueError(f"{path}: llm 필드에 누락 키 {missing}")
    return rows


def load_direct(path):
    """직접 예측(zero/few) 결과 → 세션 예측, 사람 예측(다수결)."""
    by_sid, by_pid = {}, defaultdict(list)
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        sid = normalize_sid(r["session_id"])
        by_sid[sid] = r.get("pred")
        by_pid[str(r["person_id"]).upper()].append(r.get("pred"))
    person_pred = {p: Counter(v).most_common(1)[0][0] for p, v in by_pid.items()}
    return by_sid, person_pred


def mcnemar(a_corr, b_corr):
    a_corr = np.asarray(a_corr, bool); b_corr = np.asarray(b_corr, bool)
    b = int(np.sum(a_corr & ~b_corr)); c = int(np.sum(~a_corr & b_corr))
    if b + c == 0:
        return b, c, 1.0
    if b + c < 25:
        return b, c, binomtest(min(b, c), b + c, 0.5).pvalue
    return b, c, chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1)


def bh(pvals):
    p = np.array(pvals); order = p.argsort(); m = len(p); r = np.empty(m)
    for rank, i in enumerate(order, 1):
        r[i] = p[i] * m / rank
    s = r[order][::-1]; s = np.minimum.accumulate(s)[::-1]
    out = np.empty(m); out[order] = s
    return np.clip(out, 0, 1)


def pipe():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))


# ─────────────────────────────────────────────────────────────
# LR 계열 공통 평가 (Factor 28d / Direct4 4d 모두 여기로)
# ─────────────────────────────────────────────────────────────
def eval_lr(rows, feat_keys, labels, folds, tag=""):
    """GroupKFold OOF 예측 → 세션/사람 성능 + 예측 사전 반환."""
    n = len(rows)
    X = np.array([[r["llm"][f] for f in feat_keys] for r in rows], float)
    y = np.array([r["class"] for r in rows])
    grp = np.array([r["person_id"] for r in rows])
    sids = [r["session_id"] for r in rows]

    def macro(yt, yp):
        return f1_score(yt, yp, average="macro", labels=labels, zero_division=0)

    oof = np.zeros((n, len(labels))); fold_f1 = []
    for tr, te in GroupKFold(folds).split(X, y, grp):
        p = pipe().fit(X[tr], y[tr]); cls = list(p.classes_)
        oof[te] = p.predict_proba(X[te])[:, [cls.index(c) for c in labels]]
        fold_f1.append(macro(y[te], np.array(labels)[oof[te].argmax(1)]))
    oof_pred = np.array([labels[i] for i in oof.argmax(1)])

    # 사람 정답/예측 (soft voting)
    pc = defaultdict(set)
    for r in rows:
        pc[r["person_id"]].add(r["class"])
    person_true = {p: (cs - {"NORMAL"}).pop() if (cs - {"NORMAL"}) else "NORMAL"
                   for p, cs in pc.items()}
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["person_id"]].append(i)
    pids = sorted(by)                                  # ★ 사람 순서도 고정
    p_yt = np.array([person_true[p] for p in pids])
    p_yp = np.array([labels[oof[by[p]].mean(0).argmax()] for p in pids])

    if tag:
        print(f"\n===== [{tag}] =====")
        print(f"① fold macro-F1: {np.mean(fold_f1):.3f} ± {np.std(fold_f1):.3f}  "
              f"(folds: " + ", ".join(f"{v:.3f}" for v in fold_f1) + ")")
        print(f"   [세션] macro-F1 {macro(y, oof_pred):.3f} · acc {accuracy_score(y, oof_pred):.3f}")
        print(f"   [사람] macro-F1 {macro(p_yt, p_yp):.3f} · acc {accuracy_score(p_yt, p_yp):.3f}")
        cm = confusion_matrix(p_yt, p_yp, labels=labels)
        pcf1 = f1_score(p_yt, p_yp, average=None, labels=labels, zero_division=0)
        print("   범주별 F1: " + "  ".join(f"{c[:3]}={v:.3f}" for c, v in zip(labels, pcf1)))

        # ── (논문 4.1) 3-클래스(정상 제외) macro-F1 — 사후 평가, 재학습 아님 ──
        # 4-클래스로 학습·예측한 결과를 그대로 두고, 정상인 사람만 평가에서 제외.
        # 정상으로 오분류된 장애 케이스는 labels3 기준 F1에 recall 하락으로 정직히 반영됨.
        if "NORMAL" in labels:
            mask = p_yt != "NORMAL"
            labels3 = [l for l in labels if l != "NORMAL"]
            f1_3 = f1_score(p_yt[mask], p_yp[mask],
                            average="macro", labels=labels3, zero_division=0)
            print(f"   [사람·3클래스(정상제외)] macro-F1 {f1_3:.3f}  (n={int(mask.sum())})")

    return dict(X=X, y=y, grp=grp, sids=sids, oof=oof, oof_pred=oof_pred,
                pids=pids, person_true=person_true, p_yt=p_yt, p_yp=p_yp,
                fold_f1=np.array(fold_f1))


def boot_ci(rng, yt, yp, labels, nboot):
    def macro(a, b):
        return f1_score(a, b, average="macro", labels=labels, zero_division=0)
    yt, yp = np.array(yt), np.array(yp); idx = np.arange(len(yt))
    f1s, acs = [], []
    for _ in range(nboot):
        s = rng.choice(idx, len(idx), replace=True)
        f1s.append(macro(yt[s], yp[s])); acs.append(accuracy_score(yt[s], yp[s]))
    return (np.percentile(f1s, [2.5, 97.5]), np.percentile(acs, [2.5, 97.5]),
            macro(yt, yp), accuracy_score(yt, yp))


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True, help="session_llm_{tag}.jsonl (Factor 28d)")
    ap.add_argument("--direct4", default=None, help="session_direct4_{tag}.jsonl (4d → LR)")
    ap.add_argument("--zero", default=None)
    ap.add_argument("--few", default=None)
    ap.add_argument("--sig-out", default=None)
    ap.add_argument("--pred-out-dir", default=None, help="내담자 예측 jsonl 저장 디렉토리")
    ap.add_argument("--tag", default="", help="출력 파일명 태그 (예: 14b)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--nboot-metric", type=int, default=2000)
    ap.add_argument("--nboot-coef", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Factor (28d) ──
    rows = load_rows(args.llm, SYMPTOM28)
    labels = sorted(set(r["class"] for r in rows))
    print(f"세션 {len(rows)} · 사람 {len(set(r['person_id'] for r in rows))} · 클래스 {labels}")
    fac = eval_lr(rows, SYMPTOM28, labels, args.folds, tag="Factor(28d)")

    # ── ② bootstrap CI (Factor) ──
    (f1lo, f1hi), (aclo, achi), f1v, acv = boot_ci(rng, fac["y"], fac["oof_pred"],
                                                   labels, args.nboot_metric)
    print(f"\n② [세션] macro-F1 {f1v:.3f} [{f1lo:.3f}, {f1hi:.3f}] · acc {acv:.3f} [{aclo:.3f}, {achi:.3f}]")
    (pf1lo, pf1hi), (paclo, pachi), pf1v, pacv = boot_ci(rng, fac["p_yt"], fac["p_yp"],
                                                         labels, args.nboot_metric)
    print(f"   [사람] macro-F1 {pf1v:.3f} [{pf1lo:.3f}, {pf1hi:.3f}] · acc {pacv:.3f} [{paclo:.3f}, {pachi:.3f}]")

    # ── ③ 계수 그룹-bootstrap + BH-FDR (Factor만) ──
    X, y, grp = fac["X"], fac["y"], fac["grp"]
    uniq = sorted(set(grp))                             # ★ 그룹 순서 고정
    gidx = {g: np.where(grp == g)[0] for g in uniq}
    Bc = np.zeros((args.nboot_coef, len(labels), len(SYMPTOM28)))
    for b in range(args.nboot_coef):
        sg = rng.choice(len(uniq), len(uniq), replace=True)
        ii = np.concatenate([gidx[uniq[k]] for k in sg])
        p = pipe().fit(X[ii], y[ii]); cls = list(p.classes_)
        Bc[b] = p.named_steps["logisticregression"].coef_[[cls.index(c) for c in labels]]
    allp, meta = [], []
    for ci in range(len(labels)):
        for fi in range(len(SYMPTOM28)):
            frac = np.mean(Bc[:, ci, fi] > 0)
            allp.append(2 * min(frac, 1 - frac)); meta.append((ci, fi))
    q = bh(allp); qmap = {meta[k]: q[k] for k in range(len(q))}

    print("\n③ 클래스별 유의 기여 factor (계수 95% CI, BH-FDR<0.05, 양의 상위)")
    for ci, c in enumerate(labels):
        items = []
        for fi, f in enumerate(SYMPTOM28):
            m = Bc[:, ci, fi].mean(); lo, hi = np.percentile(Bc[:, ci, fi], [2.5, 97.5])
            if m > 0 and qmap[(ci, fi)] < 0.05 and lo > 0:     # sig.json과 동일 기준
                items.append((m, f, lo, hi))
        items.sort(reverse=True)
        print(f"  {c:11}: " + ", ".join(f"{f}({m:+.2f}[{lo:+.2f},{hi:+.2f}])"
                                        for m, f, lo, hi in items[:5]))

    if args.sig_out:
        coef_mean = {c: {SYMPTOM28[fi]: float(Bc[:, ci, fi].mean())
                         for fi in range(len(SYMPTOM28))} for ci, c in enumerate(labels)}
        q_json = {c: {SYMPTOM28[fi]: float(qmap[(ci, fi)])
                      for fi in range(len(SYMPTOM28))} for ci, c in enumerate(labels)}
        sig_json = {c: {SYMPTOM28[fi]: bool(np.percentile(Bc[:, ci, fi], 2.5) > 0
                                            and qmap[(ci, fi)] < 0.05)
                        for fi in range(len(SYMPTOM28))} for ci, c in enumerate(labels)}
        with open(args.sig_out, "w", encoding="utf-8") as f:
            json.dump({"sig": sig_json, "q": q_json, "coef_mean": coef_mean},
                      f, ensure_ascii=False, indent=2)
        print(f"   sig.json 저장: {args.sig_out}")

    # ── ④ McNemar: Factor vs 각 베이스라인 ──
    print("\n④ McNemar (Factor vs baseline)")
    true_s = {r["session_id"]: r["class"] for r in rows}
    fac_sess = {s: p for s, p in zip(fac["sids"], fac["oof_pred"])}
    fac_person = {p: yp for p, yp in zip(fac["pids"], fac["p_yp"])}
    person_true = fac["person_true"]

    # (6) Holm-Bonferroni 보정을 위해 사람 단위 p값을 수집
    person_pvals = {}

    def report(name, b_sid, b_person):
        common_s = [s for s in fac["sids"] if s in b_sid]
        fc = np.array([fac_sess[s] == true_s[s] for s in common_s])
        bc = np.array([b_sid[s] == true_s[s] for s in common_s])
        b, c, pv = mcnemar(fc, bc)
        print(f"  [세션] Factor vs {name:10}: n={len(common_s)} F✓={b}, {name}✓={c}, "
              f"p={pv:.2e} (acc {fc.mean():.3f} vs {bc.mean():.3f})")
        common_p = [p for p in fac["pids"] if p in b_person]
        fcp = np.array([fac_person[p] == person_true[p] for p in common_p])
        bcp = np.array([b_person[p] == person_true[p] for p in common_p])
        b, c, pv = mcnemar(fcp, bcp)
        person_pvals[name] = pv                        # ★ 사람 단위 p 수집
        print(f"  [사람] Factor vs {name:10}: n={len(common_p)} F✓={b}, {name}✓={c}, "
              f"p={pv:.2e} (acc {fcp.mean():.3f} vs {bcp.mean():.3f})")

    # zero / few: 직접 예측
    for name, path in [("zero-shot", args.zero), ("few-shot", args.few)]:
        if path and os.path.exists(path):
            b_sid, b_person = load_direct(path)
            report(name, b_sid, b_person)
        else:
            print(f"  {name}: 파일 없음")

    # Direct4: Factor와 동일 파이프라인으로 학습해 예측 생성
    d4 = None
    if args.direct4 and os.path.exists(args.direct4):
        rows4 = load_rows(args.direct4, CATEGORIES4)
        if [r["session_id"] for r in rows4] != fac["sids"]:
            print("  ⚠ Direct4 세션 집합이 Factor와 다릅니다 — 교집합으로 비교됩니다.")
        d4 = eval_lr(rows4, CATEGORIES4, labels, args.folds, tag="Direct4(4d→LR)")
        d4_sid = {s: p for s, p in zip(d4["sids"], d4["oof_pred"])}
        d4_person = {p: yp for p, yp in zip(d4["pids"], d4["p_yp"])}
        report("Direct4", d4_sid, d4_person)
        # Direct4 사람 단위 CI (Table 3 캡션용)
        (l, h), _, v, _ = boot_ci(rng, d4["p_yt"], d4["p_yp"], labels, args.nboot_metric)
        print(f"   [사람] Direct4 macro-F1 {v:.3f} [{l:.3f}, {h:.3f}]")
    else:
        print("  Direct4: 파일 없음")

    # ── (6) Holm-Bonferroni 보정 (사람 단위 McNemar, 방법 비교) ──
    if person_pvals:
        print("\n⑤ Holm-Bonferroni 보정 (사람 단위, 방법 간 비교)")
        items = sorted(person_pvals.items(), key=lambda kv: kv[1])   # p 오름차순
        m = len(items)
        prev = 0.0
        for rank, (name, pv) in enumerate(items):       # rank: 0..m-1
            adj = pv * (m - rank)                        # 가장 작은 p에 ×m
            adj = min(adj, 1.0)
            adj = max(adj, prev)                         # 단조성 보정
            prev = adj
            mark = "✓" if adj < 0.05 else "✗"
            print(f"  Factor vs {name:10}: raw p={pv:.2e} → Holm p={adj:.2e}  {mark}")

    # ── 내담자 예측 저장 (감사·재현용) ──
    if args.pred_out_dir:
        os.makedirs(args.pred_out_dir, exist_ok=True)
        def dump(pids_, yt_, yp_, name):
            path = os.path.join(args.pred_out_dir, f"pred_person_{name}_{args.tag}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for p, t, pr in zip(pids_, yt_, yp_):
                    f.write(json.dumps({"person_id": p, "true": str(t), "pred": str(pr)},
                                       ensure_ascii=False) + "\n")
            print(f"   저장: {path}")
        dump(fac["pids"], fac["p_yt"], fac["p_yp"], "factor")
        if d4:
            dump(d4["pids"], d4["p_yt"], d4["p_yp"], "direct4")


if __name__ == "__main__":
    main()