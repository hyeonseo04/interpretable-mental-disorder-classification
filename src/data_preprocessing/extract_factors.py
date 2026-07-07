#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 요인 추출 (B2 제안 모델의 핵심 단계)
- 입력: data_loader.py가 만든 sessions.jsonl (각 줄에 text/class/gold_factors 등)
- 처리: 세션 텍스트(내담자 발화) → 48개 요인을 0~3으로 추출 (요인당 1회, 결정론적)
- 출력: sessions_with_llm.jsonl (llm_factors_raw 0~3 / llm_factors threshold≥2 추가)

실행 (예):
    python3 extract_factors.py sessions.jsonl sessions_with_llm.jsonl \
        --model Qwen/Qwen2.5-32B-Instruct --tp 2 --limit 0
    # --limit N : 앞 N개 세션만(테스트). 0이면 전체
필요: pip install vllm
GPU: 32B는 A6000×2(tp=2) 또는 H100(tp=1)
"""
import sys, json, re, argparse

# ---- 48개 요인 추출 질문 (v2 스펙 확정판) ----
QUESTIONS = {
 # 증상요인
 "depressive_mood":"내담자가 슬픔·공허감·절망감 등 가라앉은 기분을 표현하는 정도는?",
 "worthlessness":"자신이 가치 없거나 쓸모없다고 느끼는 정도는?",
 "guilt":"과도한 죄책감·자책을 표현하는 정도는?",
 "impaired_cognition":"사고력·판단력·결정 능력이 떨어졌다고 호소하는 정도는?",
 "suicidal":"죽음·자해·자살에 대한 생각을 표현하는 정도는?",
 "anhedonia":"평소 즐기던 활동에 대한 흥미·즐거움이 줄어든 정도는?",
 "psychomotor_changes":"행동·말이 느려지거나 반대로 안절부절못하는 정도는?",
 "weight_appetite":"식욕·체중의 뚜렷한 변화를 보이는 정도는?",
 "sleep_disturbance":"불면·과다수면 등 수면 문제를 호소하는 정도는?",
 "fatigue":"피로감·에너지 저하를 호소하는 정도는?",
 "anxiety_mood":"막연한 걱정·초조·불안한 기분을 표현하는 정도는?",
 "derealization":"현실·자신·주변이 비현실적으로 느껴진다고 표현하는 정도는?",
 "perceived_loss_of_control":"상황·자신을 통제할 수 없다고 느끼는 정도는?",
 "anxiety_control":"불안·걱정을 스스로 멈추거나 조절하기 어렵다고 호소하는 정도는?",
 "concentration":"집중·주의 유지가 어렵다고 호소하는 정도는?",
 "avoidance":"특정 상황·사회적 자리를 회피하는 정도는?",
 "physical_symptoms":"두근거림·떨림·답답함 등 불안 관련 신체 증상을 호소하는 정도는?",
 "irritability":"사소한 일에 짜증·화가 나는 등 과민한 정도는?",
 "loss_of_control":"특정 행동을 의도대로 조절하지 못하는 정도는?",
 "craving":"특정 대상·행동에 대한 강한 욕구·갈망을 표현하는 정도는?",
 "lying":"자신의 행동을 숨기거나 거짓말하는 정도는?",
 "tolerance":"같은 효과를 위해 양·빈도를 늘려야 한다고 표현하는 정도는?",
 "withdrawal":"중단 시 금단 증상(불안·떨림 등)을 표현하는 정도는?",
 "salience":"특정 행동이 삶의 중심이 되어 다른 활동보다 우선시되는 정도는?",
 "resource_investment":"특정 행동에 시간·돈·에너지를 과도하게 투입하는 정도는?",
 "daily_functioning":"일상생활·자기관리 기능이 저하된 정도는?",
 "social_problems":"특정 행동으로 대인관계·사회적 문제가 발생한 정도는?",
 "negative_consequences":"특정 행동으로 신체·심리·경제적 부정적 결과를 겪는 정도는?",
 "self_control":"스스로 행동(예: 음주·사용)을 통제·절제하지 못하는 정도는?",
 # 위험요인
 "trauma_experience":"과거 외상적 경험을 언급하는 정도는?",
 "negative_self-image":"자신에 대한 부정적 인식·평가를 표현하는 정도는?",
 "emotional_regulation":"정서(분노 등)를 조절·다스리는 데 어려움을 보이는 정도는?",
 "motivation_for_change":"변화·개선하려는 의지·동기를 표현하는 정도는?",
 "irrational_beliefs":"비합리적이거나 왜곡된 신념을 표현하는 정도는?",
 "unrealistic_recovery_expectations":"회복에 대해 비현실적 기대를 보이는 정도는?",
 "coping":"스트레스·문제에 대한 (특히 미숙·반응적) 대처 양상이 드러나는 정도는?",
 "lifestyle":"불규칙·부적응적 생활습관을 보이는 정도는?",
 "family_history":"정신건강 문제·중독의 가족력을 언급하는 정도는?",
 "underlying_physical_condition":"신체적 기저질환을 언급하는 정도는?",
 "history_of_mental_illness":"과거 정신질환 이력을 언급하는 정도는?",
 "stressful_event":"최근·지속적 스트레스 사건을 언급하는 정도는?",
 "social_support":"사회적 지지(가족·친구)가 부족하거나 지지받지 못하는 정도는?",
 "social_resources":"활용 가능한 자원(경제·환경)이 부족하거나 어려움을 겪는 정도는?",
 "reward_sensitivity":"보상·즉각적 만족에 민감하게 반응하는 정도는?",
 "self_management":"일상에서 스스로를 관리·조절하는 데 어려움을 보이는 정도는?",
 "social_norms":"주변(또래·가족)이 특정 행동을 당연시·함께하는 등 환경이 그 행동을 정상화·조장하는 정도는?",
 "accepting_attitude":"자신의 문제 행동(예: 음주·흡연)을 긍정·정당화·합리화하는 정도는?",
 "opportunity":"특정(중독) 행동에 노출·접근할 기회가 있는 정도는?",
}
FACTORS = list(QUESTIONS.keys())
assert len(FACTORS) == 48
THRESHOLD = 2
SYS = "당신은 심리상담 텍스트를 분석하는 임상 전문가입니다. 주어진 내담자 발화만 근거로 질문에 0~3 정수로만 답합니다."

def user_prompt(text, question):
    # 세션 텍스트를 앞(공유 prefix)에, 질문을 끝에 → vLLM prefix 캐시 활용
    return (f"[상담 내용]\n{text}\n[상담 내용 끝]\n\n"
            f"위 내담자 발화를 근거로 다음 질문에 답하세요.\n질문: {question}\n\n"
            "0에서 3 사이의 정수 점수로만 답하세요.\n"
            "- 0 = 해당 요인이 나타나지 않음\n"
            "- 1 = 유추해야 알 수 있는 수준으로만 암시됨\n"
            "- 2 = 모호한 형태로 나타남\n"
            "- 3 = 명시적으로 뚜렷하게 나타남\n"
            "점수 숫자 하나만 반환하세요.")

def parse_score(s):
    m = re.search(r"[0-3]", s)
    return int(m.group()) if m else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_model_len", type=int, default=8192)
    args = ap.parse_args()

    sessions = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    if args.limit: sessions = sessions[:args.limit]
    print(f"세션 {len(sessions)}개 × 요인 {len(FACTORS)} = 프롬프트 {len(sessions)*len(FACTORS)}개")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, enable_prefix_caching=True,
              gpu_memory_utilization=0.92)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4)

    # 세션별로 묶어 prefix 캐시 적중 ↑ (같은 text의 48개 질문 연속 배치)
    prompts, idx = [], []   # idx[i] = (session_i, factor)
    for si, s in enumerate(sessions):
        text = s.get("text","")
        for fac in FACTORS:
            msgs = [{"role":"system","content":SYS},
                    {"role":"user","content":user_prompt(text, QUESTIONS[fac])}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            idx.append((si, fac))

    outs = llm.generate(prompts, sp)
    scores = [parse_score(o.outputs[0].text) for o in outs]

    for s in sessions:
        s["llm_factors_raw"] = {f:0 for f in FACTORS}
    for (si, fac), sc in zip(idx, scores):
        sessions[si]["llm_factors_raw"][fac] = sc
    for s in sessions:
        s["llm_factors"] = {f:(v if v>=THRESHOLD else 0)
                            for f,v in s["llm_factors_raw"].items()}

    with open(args.out, "w", encoding="utf-8") as w:
        for s in sessions:
            w.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[저장] {args.out}")

    # sanity: 클래스별 LLM ≥2 활성 요인 수 평균 (골드와 비슷한 패턴이어야)
    from collections import defaultdict
    agg = defaultdict(list)
    for s in sessions:
        agg[s.get("class","?")].append(sum(1 for v in s["llm_factors"].values() if v>0))
    print("=== LLM threshold≥2 세션당 활성 요인 수(평균) ===")
    for c,v in agg.items(): print(f"  {c}: {sum(v)/len(v):.1f}")

if __name__ == "__main__":
    main()