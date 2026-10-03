# -*- coding: utf-8 -*-
"""
prompts.py — 4개 방법의 프롬프트 빌더 + 출력 파서.

- factor28 : 제안 방법 — 28개 증상요인 강도(0~3) JSON
- direct4  : Direct-to-LR 베이스라인 — 4개 범주 강도(0~3) JSON
- zeroshot / fewshot : LLM 직접 분류 베이스라인 — {"class": ...}
전사는 자르지 않고 전체를 넣는다 (data.py).
파서는 실패 시 0 / NORMAL 로 채우되, 실패 여부(status)를 함께 반환해 결과 파일에 기록한다.
"""
import json
import re

from .constants import CATEGORIES4, CLASSES, SYMPTOM28

_DIRECT_CLASSES = ["DEPRESSION", "ANXIETY", "ADDICTION", "NORMAL"]  # zero/few 프롬프트·폴백 순서

# ════════════════════════════════════════════════════════════════
# factor28 — 제안 방법
# ════════════════════════════════════════════════════════════════
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

FACTOR_SCALE = ("0 = 증상이 나타나지 않음\n1 = 약하게 나타남\n"
                "2 = 어느 정도 뚜렷이 나타남\n3 = 강하게/심하게 나타남")

FACTOR_SYSTEM = ("당신은 한국어 심리상담 대화를 분석하는 임상 전문가입니다. "
                 "한 상담 세션에서 내담자가 보인 아래 28개 증상요인이 "
                 "'얼마나 강하게 나타나는지(증상의 정도)'를 0~3으로 평가합니다. "
                 "누구나 겪는 일시적·일반적 표현은 낮게(0~1), "
                 "임상적으로 뚜렷하고 반복적으로 드러나는 증상만 높게(2~3) 평가하세요.")


def build_factor28(transcript: str):
    block = "\n".join(f"- {f}: {FACTOR_DESC[f]}" for f in SYMPTOM28)
    user = (f"[내담자 발화 모음]\n{transcript}\n\n[평가 척도]\n{FACTOR_SCALE}\n\n"
            f"[증상요인 28개]\n{block}\n\n"
            "위 세션에서 내담자가 보인 28개 요인의 점수를 매기세요. 설명 없이 JSON 객체 하나만 출력하세요.\n"
            '형식 예: {"depressive_mood": 0, "anxiety_mood": 2, ... (28개 모두 포함)}')
    return [{"role": "system", "content": FACTOR_SYSTEM},
            {"role": "user", "content": user}]


# ════════════════════════════════════════════════════════════════
# direct4 — Direct-to-LR 베이스라인
# ════════════════════════════════════════════════════════════════
CATEGORY_DESC = {
    "depression": "우울 경향 — 지속적 슬픔·무기력·흥미상실 등 우울 관련 신호의 전반적 강도",
    "anxiety": "불안 경향 — 과도한 걱정·초조·긴장·회피 등 불안 관련 신호의 전반적 강도",
    "addiction": "중독 경향 — 조절 실패·갈망·집착 등 중독 관련 신호의 전반적 강도",
    "normal": "정상 상태 — 임상적 문제 없이 안정적으로 기능하는 상태의 전반적 강도",
}

DIRECT4_SCALE = ("0 = 나타나지 않음\n1 = 약하게 나타남\n"
                 "2 = 어느 정도 뚜렷이 나타남\n3 = 강하게/심하게 나타남")

DIRECT4_SYSTEM = ("당신은 한국어 심리상담 대화를 분석하는 임상 전문가입니다. "
                  "한 상담 세션에서 내담자가 보인 우울·불안·중독 경향과 정상 상태가 "
                  "'얼마나 강하게 나타나는지'를 각각 0~3으로 평가합니다. "
                  "누구나 겪는 일시적·일반적 표현은 낮게(0~1), "
                  "임상적으로 뚜렷하고 반복적으로 드러나는 경향만 높게(2~3) 평가하세요.")


def build_direct4(transcript: str):
    block = "\n".join(f"- {c}: {CATEGORY_DESC[c]}" for c in CATEGORIES4)
    user = (f"[내담자 발화 모음]\n{transcript}\n\n[평가 척도]\n{DIRECT4_SCALE}\n\n"
            f"[평가 항목 4개]\n{block}\n\n"
            "위 세션에 대해 4개 항목의 점수를 매기세요. 설명 없이 JSON 객체 하나만 출력하세요.\n"
            '형식 예: {"depression": 2, "anxiety": 1, "addiction": 0, "normal": 1}')
    return [{"role": "system", "content": DIRECT4_SYSTEM},
            {"role": "user", "content": user}]


# ════════════════════════════════════════════════════════════════
# zero-shot / few-shot — LLM 직접 분류 베이스라인
# ════════════════════════════════════════════════════════════════
DIRECT_SYSTEM = ("당신은 한국어 심리상담 대화를 분석하는 임상 전문가입니다. "
                 "내담자의 발화를 보고 가장 적합한 범주 하나로 분류하세요.")

# few-shot 예시: 실제 세션이 아닌 **연구진이 작성한 합성 발화** (범주별 1개 × 각 3문장, 4-shot).
# 데이터에서 뽑지 않으므로 폴드 누설이 구조적으로 불가능하다. 길이는 예시당 108~124자.
FEWSHOT_EXAMPLES = {
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


def _direct_user(transcript: str) -> str:
    return (f"[내담자 발화 모음]\n{transcript}\n\n"
            "이 내담자를 다음 네 범주 중 하나로 분류하세요:\n"
            "- DEPRESSION (우울)\n- ANXIETY (불안)\n- ADDICTION (중독)\n- NORMAL (정상)\n\n"
            '설명 없이 JSON 객체 하나만 출력하세요. 예: {"class": "DEPRESSION"}')


def build_zeroshot(transcript: str):
    return [{"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": _direct_user(transcript)}]


def build_fewshot(transcript: str):
    msgs = [{"role": "system", "content": DIRECT_SYSTEM}]
    for cls in _DIRECT_CLASSES:                       # 순서: DEP → ANX → ADD → NOR
        ex_tr = "\n".join(FEWSHOT_EXAMPLES[cls])
        msgs.append({"role": "user", "content": _direct_user(ex_tr)})
        msgs.append({"role": "assistant", "content": json.dumps({"class": cls}, ensure_ascii=False)})
    msgs.append({"role": "user", "content": _direct_user(transcript)})
    return msgs


# ════════════════════════════════════════════════════════════════
# 파서 — 실패 시 기본값 + 상태 반환
# ════════════════════════════════════════════════════════════════
def _first_json_obj(text: str):
    m = re.search(r"\{.*\}", text, re.S)          # 첫 { 부터 마지막 } 까지 (greedy)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def parse_score_vector(text: str, keys):
    """0~3 정수 벡터 파싱. 실패 키는 0.
    returns (scores, status, n_missing) — status ∈ ok / partial / no_json"""
    obj = _first_json_obj(text)
    if obj is None:
        return {k: 0 for k in keys}, "no_json", len(keys)
    out, n_missing = {}, 0
    for k in keys:
        if k not in obj:
            out[k] = 0; n_missing += 1; continue
        try:
            out[k] = max(0, min(3, int(obj[k])))
        except Exception:
            out[k] = 0; n_missing += 1
    return out, ("ok" if n_missing == 0 else "partial"), n_missing


def parse_class(text: str):
    """JSON class → 실패 시 본문 키워드 폴백 → None.
    returns (pred_or_None, status) — status ∈ ok / keyword_fallback / fail"""
    obj = _first_json_obj(text) or {}
    c = str(obj.get("class", "")).upper()
    if c in CLASSES:
        return c, "ok"
    for cand in _DIRECT_CLASSES:
        if cand in text.upper():
            return cand, "keyword_fallback"
    return None, "fail"


BUILDERS = {"factor28": build_factor28, "direct4": build_direct4,
            "zeroshot": build_zeroshot, "fewshot": build_fewshot}
