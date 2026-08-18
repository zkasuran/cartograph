"""Tests for the ontology deep-search query engine.

Fast, deterministic tests run by default (helpers + one real person query over the
live graph). Tests that call the LLM planner are gated behind ONTO_LLM_TESTS=1 so
`pytest -q` stays fast and offline-friendly. Graph tests skip cleanly when no
HydraDB node is reachable.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from onto import query  # noqa: E402
from onto.graph import AUTHORED, PARTICIPATED, OntologyGraph  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ONTO = json.loads((REPO / "data" / "ontology.json").read_text(encoding="utf-8"))


def _node_up() -> bool:
    base = os.environ.get("HYDRA_HTTP", "http://127.0.0.1:8443")
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/healthz", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


requires_node = pytest.mark.skipif(not _node_up(), reason="HydraDB node not reachable")
requires_llm = pytest.mark.skipif(
    os.environ.get("ONTO_LLM_TESTS") != "1", reason="set ONTO_LLM_TESTS=1 to run LLM tests")


def _known_person_q() -> dict:
    for q in ONTO["questions"]:
        if (q["answerable"] and q["type"] == "person"
                and "authors and key reviewers" in q["question"] and q["ground_truth"]):
            return q
    raise AssertionError("no 'authors and key reviewers' person question in the dataset")


# ---------------------------------------------------------------- fast, no network
def test_resolve_product():
    assert query.resolve_product(None, "the ActionGenie product") == "ActionGenie"
    assert query.resolve_product("anomaly force") == "AnomalyForce"
    assert query.resolve_product("SearchForce") == "SearchForce"
    assert query.resolve_product("not a real thing", "no product named here") is None


def test_question_detectors():
    q = "Find employee IDs of Product Managers who reviewed the Market Research Report for ActionGenie?"
    assert query._detect_role(q) == "Product Manager"
    assert query._detect_doc_type(q) == "Market Research Report"
    assert query._detect_pr_status("PRs for X that were not approved?") == "not_approved"
    assert query._detect_pr_status("the approved PRs for X") == "approved"
    assert query._detect_pr_status("all PRs for X") == "any"
    assert query._superlative_prs(
        "engineer with the highest number of approved feature development PRs") == "approved"
    assert query._superlative_prs("who reviewed the report") is None


def test_herb_pr_join_and_status():
    prs = query._herb_prs("AnomalyForce")
    assert prs, "expected salesforce feature PRs for AnomalyForce"
    assert all("github.com/salesforce/" in p["link"] for p in prs)
    assert any(p["approved"] for p in prs) and any(not p["approved"] for p in prs)


# ---------------------------------------------------------------- live graph, no LLM
@requires_node
def test_person_query_returns_overlapping_eids():
    """Real end-to-end person query over the live graph (the production person
    path, LLM planner substituted by a deterministic product + keyword extraction):
    a known person question over a known product returns a non-empty eid set that
    intersects its ground truth."""
    q = _known_person_q()
    product = query.resolve_product(None, q["question"])
    assert product is not None
    keywords = list(query._tokens(q["question"]))
    graph = OntologyGraph()
    eids = query._answer_person(q["question"], graph, product, keywords, model=None)
    assert eids, "expected a non-empty eid set"
    assert set(eids) & set(q["ground_truth"]), "returned eids must overlap ground truth"


@requires_node
def test_role_attributed_doc_review_abstains():
    """A role-tagged document review has no graph evidence, so the person path
    returns empty (the abstain signal), even though the review meeting had people."""
    q = next(x for x in ONTO["questions"]
             if not x["answerable"] and "Product Managers who reviewed" in x["question"])
    product = query.resolve_product(None, q["question"])
    graph = OntologyGraph()
    eids = query._answer_person(q["question"], graph, product, [], model=None)
    assert not eids


# ---------------------------------------------------------------- full pipeline (LLM)
@requires_node
@requires_llm
def test_answer_end_to_end_person():
    q = _known_person_q()
    graph = OntologyGraph()
    result = query.answer(q["question"], graph)
    assert result["type"] == "person" and result["product"]
    assert result["abstain"] is False and result["answer"]
    assert set(result["answer"]) & set(q["ground_truth"])


# ---------------------------------------------------------------- answer() with the
# LLM planner replaced (fast, deterministic, no network LLM). The seam is the plan
# function query._plan, the only LLM call answer() makes on these paths; the rest is
# graph traversal + entity resolution. answer() reads _plan as a module global, so
# monkeypatch.setattr(query, "_plan", ...) is picked up by the running answer().
def _person_q(product: str, doc_type: str) -> dict:
    for q in ONTO["questions"]:
        if (q["answerable"] and q["type"] == "person" and q["product"] == product
                and doc_type in q["question"] and "authors and key reviewers" in q["question"]
                and q["ground_truth"]):
            return q
    raise AssertionError(f"no {doc_type!r} person question for {product}")


@requires_node
def test_answer_person_path_via_monkeypatched_plan(monkeypatch):
    """Full answer() person path with the LLM planner replaced by a fixed plan: a
    real product + person type + topic keywords resolves to real employee eids that
    intersect the question's ground truth. Pins the graph person path end to end
    without the network LLM."""
    q = _person_q("ActionGenie", "Product Vision Document")
    plan = {"product": "ActionGenie", "type": "person", "keywords": ["vision", "document"]}
    monkeypatch.setattr(query, "_plan", lambda question, model=None: plan)
    result = query.answer(q["question"], OntologyGraph())
    assert result["type"] == "person" and result["product"] == "ActionGenie"
    assert result["abstain"] is False and result["answer"]
    emp_ids = {e["eid"] for e in ONTO["employees"]}
    assert set(result["answer"]) <= emp_ids, "answer must be real employee eids"
    assert set(result["answer"]) & set(q["ground_truth"]), "answer must intersect ground truth"


@requires_node
def test_answer_abstains_on_empty_traversal(monkeypatch):
    """An empty bounded traversal means the answer is not in the corpus, so answer()
    abstains instead of inventing ids: a real product + person type but keywords that
    match no artifact yields abstain=True and an empty answer."""
    plan = {"product": "ActionGenie", "type": "person", "keywords": ["zqxjkbwpmnvxyz"]}
    monkeypatch.setattr(query, "_plan", lambda question, model=None: plan)
    result = query.answer("Who contributed to the internal initiative?", OntologyGraph())
    assert result["product"] == "ActionGenie"
    assert result["abstain"] is True
    assert result["answer"] == []


def test_answer_abstains_when_product_unresolved(monkeypatch):
    """When the plan names a product that resolves to none of the 30 real products
    (and the question names none either), answer() abstains with a null product and
    an empty answer rather than guessing. Abstains before any graph access."""
    fake = "Nonexistent Widget 9000"
    question = "list the people on the internal effort"
    assert query.resolve_product(fake, question) is None  # precondition for the abstain
    plan = {"product": fake, "type": "person", "keywords": ["people", "effort"]}
    monkeypatch.setattr(query, "_plan", lambda q, model=None: plan)
    result = query.answer(question, OntologyGraph())
    assert result["product"] is None
    assert result["abstain"] is True
    assert result["answer"] == []
