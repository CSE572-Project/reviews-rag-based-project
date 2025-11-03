import json, re, time, argparse
from pathlib import Path
from statistics import mean
from rag_chess import answer_question

def has_all(needle_list, hay):
    hay_l = hay.lower()
    return all(n.lower() in hay_l for n in needle_list)

def p_at_k_true(must_terms, context_docs):
    ctx = " ".join(d["text"].lower() for d in context_docs)
    return 1.0 if all(term.lower() in ctx for term in must_terms) else 0.0

def eval_once(item, k=6):
    t0 = time.time()
    res = answer_question(item["q"], k=k)
    dt = (time.time() - t0) * 1000
    ans = res["answer"]
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', ans) if s.strip()]
    grounded = [1 if re.search(r'\[\d+\]', s) else 0 for s in sents]
    groundedness = sum(grounded)/max(1,len(sents))
    coverage = 1.0 if has_all(item.get("must",[]), ans) else 0.0
    pk = p_at_k_true(item.get("must", []), res.get("context", []))
    return {"q": item["q"], "coverage": coverage, "groundedness": groundedness,
            "p_at_k_true": pk, "latency_ms": dt, "citations": res.get("citations", [])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--label", type=str, default="RUN")
    args = ap.parse_args()

    items = [json.loads(l) for l in Path("eval_chess_qna.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [eval_once(it, k=args.k) for it in items]

    cov = mean(r['coverage'] for r in rows)
    grd = mean(r['groundedness'] for r in rows)
    patk = mean(r['p_at_k_true'] for r in rows)
    lat = mean(r['latency_ms'] for r in rows)

    print("\n| Config | Coverage | Groundedness | True P@k | Latency (ms) |")
    print("|---|:--:|:--:|:--:|:--:|")
    print(f"| {args.label} | {cov:.2f} | {grd:.2f} | {patk:.2f} | {lat:.0f} |")

if __name__ == "__main__":
    main()
