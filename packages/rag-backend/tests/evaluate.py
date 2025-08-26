import json
import os
import re
import statistics
import time
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv

# 0. 환경변수 로드
load_dotenv()
host = os.getenv("CLIENT_HOST", "127.0.0.1")  # 요청 보낼 때는 127.0.0.1 추천
port = int(os.getenv("PORT", "8000"))
eval_api_url = f"http://{host}:{port}/api/generate"
eval_data_path = os.getenv("EVAL_DATA_PATH", "rag-evaluations.jsonl")

# 1. 평가 문항 로드 (JSONL: {"question":..., "choices":[...], "answer":"..."} per line)
def load_questions(path: str) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions

questions = load_questions(eval_data_path)

def build_prompt(q: Dict[str, Any]) -> str:
    # 모델이 포맷을 지키도록 명확한 지시 + 예시 포함
    return (
        "다음 질문에 대해 보기 중 정답을 한 개 고르고, 이유를 간단히 설명하세요.\n"
        f"[질문] {q['question']}\n"
        f"[보기] {q['choices']}\n\n"
        "반드시 아래 형식을 정확히 지키세요.\n"
        "[정답] (위 보기 중 하나 그대로)\n"
        "[이유] (간단 설명)\n"
    )

# [정답] ... [이유] ... 형식에서 정답만 robust하게 추출
ANSWER_PATTERN = re.compile(r"\[정답\]\s*(.+?)(?:\n|\r|$)", re.IGNORECASE)

def extract_answer(text: str) -> str:
    m = ANSWER_PATTERN.search(text)
    if not m:
        return ""
    return m.group(1).strip()

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def is_correct(pred: str, gold: str, choices: List[str]) -> bool:
    pred_n = normalize(pred)
    gold_n = normalize(gold)

    # 1) 완전 일치
    if pred_n == gold_n:
        return True

    # 2) 보기에 포함되는지(모델이 선택지 텍스트+부연을 같이 적는 경우 방지)
    #   - ex) "2) 파리 — 수도" 처럼 선택지 시작이 금/은/동/번호/텍스트로 시작할 수 있음
    for c in choices:
        c_n = normalize(c)
        if c_n == gold_n and c_n in pred_n:
            return True
    return False

def evaluate_via_api(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    sess = requests.Session()
    correct = 0
    latencies_ms: List[int] = []
    failures: List[Dict[str, Any]] = []

    for idx, q in enumerate(questions, 1):
        prompt = build_prompt(q)
        try:
            t0 = time.perf_counter()
            # ★ 서버 스펙에 맞는 POST JSON 바디
            resp = sess.post(
                eval_api_url,
                json={"query": prompt, "use_rag": True, "candidate_k": 10, "top_k": 4, "max_tokens": 1024},
                timeout=60,
            )
            t1 = time.perf_counter()
        except requests.RequestException as e:
            failures.append({"q": q["question"], "error": f"request_error: {e}"})
            continue

        if resp.status_code != 200:
            failures.append({"q": q["question"], "error": f"status_{resp.status_code}", "body": resp.text[:500]})
            continue

        try:
            data = resp.json()
        except Exception as e:
            failures.append({"q": q["question"], "error": f"json_error: {e}", "body": resp.text[:500]})
            continue

        print(data)
        raw_answer = data.get("response", "")
        elapsed_ms = data.get("elapsed_ms")
        if isinstance(elapsed_ms, int):
            latencies_ms.append(elapsed_ms)
        else:
            # 서버가 elapsed_ms를 안 주면 클라이언트 측에서 측정한 값 사용
            latencies_ms.append(int((t1 - t0) * 1000))

        pred = extract_answer(raw_answer)
        gold = q["answer"]
        ok = is_correct(pred, gold, q["choices"])

        # 진행 로그 (필요시 주석 처리)
        print(f"[{idx}/{len(questions)}] Q: {q['question']}")
        print(f"choices: {q['choices']}")
        print(f"model_response: {raw_answer.strip()[:300].replace('\\n', ' ')}")
        print(f"pred: {pred} | gold: {gold} | correct: {ok}\n")

        if ok:
            correct += 1
        
        time.sleep(0.1)  # 너무 빠른 요청을 피하기 위해 약간 대기

    total = len(questions)
    accuracy = correct / total if total else 0.0
    stats = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,  # 단일정답 객관식 정확도
        "latency_ms_mean": round(statistics.mean(latencies_ms), 2) if latencies_ms else None,
        "latency_ms_p50": round(statistics.median(latencies_ms), 2) if latencies_ms else None,
        "latency_ms_p95": round(statistics.quantiles(latencies_ms, n=20)[18], 2) if len(latencies_ms) >= 20 else None,
        "failures": failures,
    }
    return stats

if __name__ == "__main__":
    stats = evaluate_via_api(questions)
    print("\n==== Evaluation Result ====")
    print(f"총 문항: {stats['total']}")
    print(f"정답 수: {stats['correct']}")
    print(f"정확도 (Accuracy): {stats['accuracy']:.2%}")
    if stats["latency_ms_mean"] is not None:
        print(f"평균 지연(ms): {stats['latency_ms_mean']}")
        print(f"P50 지연(ms): {stats['latency_ms_p50']}")
        if stats['latency_ms_p95'] is not None:
            print(f"P95 지연(ms): {stats['latency_ms_p95']}")
    if stats["failures"]:
        print(f"\n실패 항목({len(stats['failures'])}) 예시: {stats['failures'][:3]}")
