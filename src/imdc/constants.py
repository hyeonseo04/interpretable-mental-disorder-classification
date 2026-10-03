# -*- coding: utf-8 -*-
"""프로젝트 전역 상수 — 모든 스크립트가 여기서만 가져다 쓴다 (정의 중복 금지)."""

# ── 데이터 ──────────────────────────────────────────────────────
CLIENT_SPEAKER = "내담자"
CLASSES = ["ADDICTION", "ANXIETY", "DEPRESSION", "NORMAL"]   # 논문 표·그림 순서 (ADD/ANX/DEP/NOR)
CLASS_SET = set(CLASSES)
SHORT = {"ADDICTION": "ADD", "ANXIETY": "ANX", "DEPRESSION": "DEP", "NORMAL": "NOR"}
DISORDERS = ["ADDICTION", "ANXIETY", "DEPRESSION"]

SYMPTOM28 = [
    "depressive_mood", "worthlessness", "guilt", "impaired_cognition", "suicidal", "anhedonia",
    "psychomotor_changes", "weight_appetite", "sleep_disturbance", "fatigue",
    "anxiety_mood", "derealization", "perceived_loss_of_control", "anxiety_control",
    "concentration", "avoidance", "physical_symptoms", "irritability",
    "loss_of_control", "craving", "lying", "tolerance", "withdrawal", "salience",
    "resource_investment", "daily_functioning", "social_problems", "negative_consequences",
]
CATEGORIES4 = ["depression", "anxiety", "addiction", "normal"]

# 데이터 규모 (논문 3.2절) — 매니페스트 생성 시 일치 여부를 검사
EXPECTED_N_PERSONS = 207
EXPECTED_N_SESSIONS = 1440
EXPECTED_N_CORRUPTED = 15

# ── 추론 조건 (논문 3.5절) ──────────────────────────────────────
MAX_MODEL_LEN = 32768
SEED = 0
DTYPE = "bfloat16"
# 모델 리비전은 Hugging Face 커밋 해시 전체로 고정 (main 이 갱신돼도 같은 가중치)
MODELS = {
    "14b": {"hf_id": "Qwen/Qwen2.5-14B-Instruct",
            "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"},
    "32b": {"hf_id": "Qwen/Qwen2.5-32B-Instruct",
            "revision": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"},
}
# method → (출력 파일 stem, max_tokens). 파일명 = {stem}_{model}{suffix}
METHODS = {
    "factor28": {"stem": "session_llm",            "max_tokens": 512},
    "zeroshot": {"stem": "session_direct_zeroshot", "max_tokens": 32},
    "fewshot":  {"stem": "session_direct_fewshot",  "max_tokens": 32},
    "direct4":  {"stem": "session_direct4",         "max_tokens": 128},
}
METHOD_LABEL = {"factor28": "Proposed (Factor-28→LR)", "direct4": "Direct-to-LR",
                "zeroshot": "Zero-shot", "fewshot": "Few-shot"}
LR_METHODS = ("factor28", "direct4")       # 점수 벡터 → LR 로 학습하는 방법
DIRECT_METHODS = ("zeroshot", "fewshot")   # LLM 이 범주를 직접 출력하는 방법

# ── 분류기 (논문 3.4·3.5절) ─────────────────────────────────────
N_FOLDS = 5
LR_C = 1.0
LR_MAX_ITER = 2000

# ── 통계 ───────────────────────────────────────────────────────
N_BOOT = 2000          # 성능 지표 부트스트랩 (내담자 클러스터)
N_BOOT_COEF = 500      # 회귀계수 그룹 부트스트랩 (Fig. 3)
MIN_POS_TABLE4 = 30    # Table 4 '검증 대상' 기준: 골드≥2 세션 수
MIN_POS_SPARSE = 20    # C1-2 / A2-4 희소 요인 기준: 골드≥1 세션 수


def stem(method: str, model: str, suffix: str = "_full") -> str:
    return f"{METHODS[method]['stem']}_{model}{suffix}"
