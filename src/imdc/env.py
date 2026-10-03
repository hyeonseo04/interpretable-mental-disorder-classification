# -*- coding: utf-8 -*-
"""
env.py — 실행 환경 기록·검증 (논문 3.5절 재현성 정보의 출처).

  python -m imdc.env            # 버전 표 출력 + 고정 버전과 다르면 exit 1
  python -m imdc.env --json     # JSON 출력
"""
import argparse
import importlib
import json
import os
import platform
import subprocess
import sys

# 논문 3.5절에 기재된 버전 — 바뀌면 안 됨
PINNED = {
    "python": "3.12.10",
    "vllm": "0.19.1",
    "torch": "2.10.0",
    "transformers": "5.11.0",
    "scikit-learn": "1.9.0",
}
# 참고용(논문 미기재) — 다르면 경고만
SOFT = {"numpy": "2.2.6", "scipy": "1.17.1"}
_IMPORT = {"scikit-learn": "sklearn"}


def _ver(pkg):
    try:
        m = importlib.import_module(_IMPORT.get(pkg, pkg))
        return getattr(m, "__version__", "?")
    except Exception:
        return None


def _base(v):
    """'2.10.0+cu128' -> '2.10.0' (빌드 태그 무시)"""
    return str(v).split('+')[0] if v else v


def gpu_info():
    out = {"gpus": [], "driver": None, "cuda_runtime": None}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        for line in q.stdout.strip().splitlines():
            name, mem, drv = [x.strip() for x in line.split(",")]
            out["gpus"].append({"name": name, "memory": mem})
            out["driver"] = drv
    except Exception:
        pass
    try:
        import torch
        out["cuda_runtime"] = torch.version.cuda
        out["torch_cuda_available"] = torch.cuda.is_available()
    except Exception:
        pass
    return out


def snapshot_of(model_path: str):
    """HF 캐시 스냅샷 경로면 커밋 해시를 돌려준다 (…/snapshots/<sha>)."""
    p = os.path.realpath(model_path)
    parts = p.split(os.sep)
    if "snapshots" in parts:
        i = parts.index("snapshots")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def collect():
    vers = {"python": platform.python_version()}
    for k in list(PINNED)[1:] + list(SOFT):
        vers[k] = _ver(k)
    return {"versions": vers, **gpu_info(), "platform": platform.platform(),
            "hostname": platform.node(), "argv": sys.argv}


def check(info, strict=True):
    bad = {k: (info["versions"].get(k), v) for k, v in PINNED.items()
           if _base(info["versions"].get(k)) != v}
    soft = {k: (info["versions"].get(k), v) for k, v in SOFT.items()
            if _base(info["versions"].get(k)) != v}
    return bad, soft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-strict", action="store_true")
    args = ap.parse_args()
    info = collect()
    bad, soft = check(info)
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=1))
    else:
        print(f"{'package':14}{'installed':>12}{'pinned':>12}")
        for k, v in PINNED.items():
            got = info["versions"].get(k)
            print(f"{k:14}{str(got):>12}{v:>12}  {'OK' if _base(got) == v else '✗ MISMATCH'}")
        for k, v in SOFT.items():
            got = info["versions"].get(k)
            print(f"{k:14}{str(got):>12}{v:>12}  {'OK' if got == v else '(참고: 다름)'}")
        print(f"GPU: {[g['name'] + ' ' + g['memory'] for g in info['gpus']]} · driver {info['driver']} "
              f"· torch CUDA {info.get('cuda_runtime')}")
    if bad and not args.no_strict:
        print(f"\n[실패] 논문 기재 버전과 다릅니다: {bad}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
