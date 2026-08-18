"""FastAPI console for the enterprise ontology deep-search.

The graph is already loaded in HydraDB; this server just runs query.answer over it
and serves the page. Structural questions resolve to graph traversals; questions
with no supporting path come back as an honest "not in the corpus".
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from onto.graph import OntologyGraph
from onto import query

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="enterprise ontology deep-search on HydraDB")
_g = OntologyGraph()


class Ask(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text()


@app.get("/api/examples")
def examples():
    o = json.load(open(ROOT / "data" / "ontology.json"))
    ans = [q for q in o["questions"] if q.get("answerable") and q.get("type") == "person"][:5]
    una = [q for q in o["questions"] if not q.get("answerable")][:3]
    return {"answerable": [q["question"] for q in ans],
            "unanswerable": [q["question"] for q in una]}


@app.post("/api/ask")
def ask(a: Ask):
    r = query.answer(a.question, _g)
    ans = r.get("answer")
    if isinstance(ans, (set, tuple)):
        ans = list(ans)
    return {"type": r.get("type"), "product": r.get("product"),
            "abstain": bool(r.get("abstain")), "answer": ans}


@app.get("/api/counts")
def counts():
    return {lbl: _g.count(lbl) for lbl in ["Employee", "Customer", "Product", "Org", "Artifact"]}
