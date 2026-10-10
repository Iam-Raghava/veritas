"""Hybrid retrieval tests: CJK tokenizer, TF-IDF fallback, RRF fusion."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "aml"))

from veritas import BeliefStore
from retrieval import _tokens, BeliefRetriever

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} {detail}")


# 1. CJK tokenizer: Chinese must produce tokens (was: zero tokens).
zh_toks = _tokens("张三是公司的首席执行官")
check("Chinese produces tokens", len(zh_toks) > 0, zh_toks)
check("Chinese bigrams present", any(t.startswith("cjk:") for t in zh_toks))
check("English still works", _tokens("Hello World") == ["hello", "world"])
check("Mixed script", len(_tokens("Acme的CEO是张三")) >= 3)

# 2. TF-IDF Chinese retrieval.
s = BeliefStore()
s.assert_belief("张三是Acme公司的首席执行官", source="公告",
                source_reliability=0.9, ground=True)
s.assert_belief("李四是Acme公司的首席财务官", source="公告",
                source_reliability=0.9, ground=True)
s.assert_belief("今天天气很好", source="sensor",
                source_reliability=0.9, ground=True)
r = BeliefRetriever(s)
res = r.search("谁是Acme的首席执行官", top_k=3)
check("Chinese TF-IDF retrieves", len(res) > 0)
if res:
    check("CEO ranks above CFO",
          "首席执行官" in res[0]["text"], res[0]["text"])

# 3. Hybrid retriever: works with or without dense model.
from retrieval_hybrid import HybridRetriever
h = HybridRetriever(s)
check("hybrid retriever constructs", True)
hres = h.search("谁是Acme的首席执行官", top_k=3)
check("hybrid search returns", len(hres) > 0)
if hres:
    check("hybrid CEO first",
          "首席执行官" in hres[0]["text"], hres[0]["text"])
    check("hybrid result has fusion marker",
          hres[0].get("fusion") == "rrf" or "score" in hres[0])
print(f"dense active: {h.dense_active}")

# 4. English paraphrase (dense should help; TF-IDF baseline).
s2 = BeliefStore()
s2.assert_belief("The chief executive officer of Acme is Alice",
                 source="board", source_reliability=0.9, ground=True)
s2.assert_belief("Acme's headquarters is in Austin",
                 source="board", source_reliability=0.9, ground=True)
h2 = HybridRetriever(s2)
pres = h2.search("who is the CEO of Acme", top_k=2)
check("paraphrase search returns", len(pres) > 0)
if pres:
    check("CEO belief found via paraphrase",
          "Alice" in pres[0]["text"], pres[0]["text"])

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
