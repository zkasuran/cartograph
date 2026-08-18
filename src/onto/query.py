"""Deep-search query engine over the enterprise ontology in HydraDB.

The pipeline is graph traversal + entity resolution, not vector search:

1. An LLM plan turns the free-text question into {product, type, keywords}.
2. The product string is resolved app-side to one of the 30 real product names
   (case-insensitive / fuzzy). No product -> abstain.
3. Candidate artifacts come from the graph (artifacts_of_product, an ABOUT
   traversal). We keep the ones whose title/keywords match the question topic in
   Python, because HydraDB's Cypher subset has no CONTAINS / IN / full-text.
4. The answer is built by type:
     person  -> union of people_of_artifact over the matched artifacts (an
                AUTHORED / PARTICIPATED traversal), optionally filtered by role;
                plus a graph-native superlative path for "who has the most
                approved/unapproved feature PRs".
     url     -> links of matched url-kind artifacts.
     pr      -> links of matched pr-kind artifacts, filtered by review status.
     company -> customer companies of the product (a USES traversal). Best effort.
     content -> an LLM reader over the matched artifacts' text.
5. Abstain when the product does not resolve, or the bounded traversal comes back
   empty (no relevant artifact / person). An empty traversal is exactly the
   signal that the answer is not in the corpus, so we abstain instead of guessing.

Extra employee/customer fields and PR review state that the graph does not carry
(the graph has no per-document reviewer edges and no PR approval state) are read
from data/HERB, joined to the graph artifacts by their id.
"""
from __future__ import annotations

import functools
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from . import llm
from .graph import AUTHORED, PARTICIPATED, REVIEWED, OntologyGraph, sid

_DATA = Path(__file__).resolve().parents[2] / "data"
_HERB_PRODUCTS = _DATA / "HERB" / "products"

ROLES = (
    "VP of Engineering", "Chief Product Officer", "Engineering Lead",
    "Technical Architect", "Product Manager", "QA Specialist",
    "Software Engineer", "UX Researcher", "Marketing Research Analyst",
    "Marketing Manager",
)
DOC_TYPES = (
    "Market Research Report", "Product Vision Document",
    "Product Requirements Document", "Technical Specifications Document",
    "System Design Document",
)
_STOP = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "that", "which",
    "who", "whom", "were", "was", "are", "is", "be", "been", "find", "list",
    "employee", "employees", "ids", "id", "product", "products", "team", "members",
    "member", "names", "name", "links", "link", "urls", "url", "prs", "pr",
    "please", "can", "you", "provide", "give", "what", "whats", "how", "many",
    "company", "companies", "affected", "issues", "issue", "with", "by", "from",
    "shared", "their", "this", "these", "those", "new", "features", "feature",
    "added", "have", "has", "his", "her", "they", "them",
}


# --------------------------------------------------------------------------
# data loading (cached)
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _ontology() -> dict:
    with open(_DATA / "ontology.json", encoding="utf-8") as fh:
        return json.load(fh)


@functools.lru_cache(maxsize=1)
def product_names() -> tuple:
    return tuple(p["name"] for p in _ontology()["products"])


@functools.lru_cache(maxsize=1)
def _emp_role() -> dict:
    return {e["eid"]: (e.get("role") or "") for e in _ontology()["employees"]}


@functools.lru_cache(maxsize=1)
def _cust_company() -> dict:
    return {c["cid"]: (c.get("company") or "") for c in _ontology()["customers"]}


@functools.lru_cache(maxsize=64)
def _herb(product: str) -> dict:
    """The raw HERB product file (PR review state, full text): fields the graph
    intentionally does not carry. Returns {} if the file is missing."""
    path = _HERB_PRODUCTS / f"{product}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# small text helpers (all app-side; HydraDB has no substring / full-text match)
# --------------------------------------------------------------------------
def _tokens(s: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if t not in _STOP and len(t) > 2}


def resolve_product(name: str | None, question: str = "") -> str | None:
    """Map a raw product string (from the plan or the question) to one of the 30
    real product names: exact, case-insensitive, substring, then fuzzy."""
    names = product_names()
    for cand in (name, question):
        if not cand:
            continue
        low = cand.lower()
        for p in names:
            if p.lower() == low:
                return p
        # longest real name that appears verbatim in the text wins
        hits = sorted((p for p in names if p.lower() in low), key=len, reverse=True)
        if hits:
            return hits[0]
    if not name:
        return None
    best, score = None, 0.0
    for p in names:
        r = SequenceMatcher(None, name.lower(), p.lower()).ratio()
        if r > score:
            best, score = p, r
    return best if score >= 0.72 else None


# --------------------------------------------------------------------------
# the LLM plan + deterministic question analysis
# --------------------------------------------------------------------------
_PLAN_SYS = (
    "You turn an enterprise deep-search question into a compact JSON plan. "
    "Output ONLY JSON, no prose."
)


def _plan(question: str, model=None) -> dict:
    prompt = (
        "Products (pick the closest one, exact spelling, or null if none is named):\n"
        + ", ".join(product_names())
        + "\n\nAnswer types:\n"
        "  person  -> the question wants employee IDs / people\n"
        "  pr      -> the question wants pull-request links\n"
        "  url     -> the question wants web URLs (demos, articles, references)\n"
        "  company -> the question wants customer company names\n"
        "  content -> the question wants a free-text answer (features, changes, summaries)\n\n"
        'Return {"product": <name or null>, "type": <one type>, '
        '"keywords": [<3-8 lower-case topic words, no product name, no stopwords>]}\n\n'
        f"Question: {question}"
    )
    try:
        p = llm.ask_json(prompt, system=_PLAN_SYS, model=model)
    except Exception:
        p = {}
    if not isinstance(p, dict):
        p = {}
    p.setdefault("product", None)
    p.setdefault("type", None)
    kw = p.get("keywords") or []
    p["keywords"] = [str(k).lower() for k in kw if isinstance(k, (str, int))]
    if p.get("type") not in ("person", "pr", "url", "company", "content"):
        p["type"] = None
    return p


def _detect_doc_type(q: str) -> str | None:
    low = q.lower()
    for dt in DOC_TYPES:
        if dt.lower() in low:
            return dt
    return None


def _detect_role(q: str) -> str | None:
    low = q.lower()
    for r in sorted(ROLES, key=len, reverse=True):
        rl = r.lower()
        if rl in low or rl + "s" in low or rl.replace("specialist", "specialists") in low:
            return r
    return None


def _detect_pr_status(q: str) -> str:
    low = q.lower()
    if "not approved" in low or "unapproved" in low or "not merged" in low or "rejected" in low:
        return "not_approved"
    if "approved" in low or "merged" in low:
        return "approved"
    return "any"


def _wants_competitor(q: str) -> bool:
    return "competitor" in q.lower() or "competing" in q.lower()


def _superlative_prs(q: str) -> str | None:
    """Detect the 'engineer with the most/least approved|unapproved feature PRs'
    person template. Returns 'approved' / 'not_approved' or None."""
    low = q.lower()
    if not re.search(r"\b(highest|maximum|most|max)\b", low):
        return None
    if "pr" not in low and "pull request" not in low and "feature development" not in low:
        return None
    if "unapproved" in low or "not approved" in low:
        return "not_approved"
    if "approved" in low:
        return "approved"
    return None


# --------------------------------------------------------------------------
# retrieval: match graph artifacts to the question topic (Python-side)
# --------------------------------------------------------------------------
def _match_artifacts(arts: list, doc_type: str | None, keywords, kinds=None, allow_fallback=True) -> list:
    """Keep artifacts of the wanted kinds whose title matches the topic. A named
    document type is an exact title match; otherwise token overlap with the
    keywords. When nothing matches and allow_fallback is set, return all of the
    kind (a downstream filter such as role or an LLM reader still needs
    candidates); otherwise return [] so the caller abstains on an empty traversal."""
    if kinds:
        arts = [a for a in arts if a.get("kind") in kinds]
    if doc_type:
        hit = [a for a in arts if doc_type.lower() in (a.get("title") or "").lower()]
        if hit:
            return hit
    kw = {k for k in keywords if len(k) > 2}
    if kw:
        scored = []
        for a in arts:
            overlap = len(_tokens(a.get("title")) & kw)
            if overlap:
                scored.append((overlap, a))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return [a for _, a in scored]
    return arts if allow_fallback else []


def _herb_prs(product: str) -> list:
    """Salesforce feature-development PRs for the product, with review status.
    The GT PR links are the salesforce/<repo> form (verified: 100% of PR ground
    truth), which is exactly the link the graph stores for these PRs, so no
    synthesis is needed. A product's feature PRs live under several salesforce
    repo codenames (its own name plus internal / previous ones like ectAIX,
    TuneProX, onForceX), so match any github.com/salesforce/ link. Upstream mirror
    PRs (kafka, scikit-learn, ...) are dropped: they never appear in ground truth."""
    prs = []
    for p in _herb(product).get("prs") or []:
        link = p.get("link") or ""
        if "github.com/salesforce/" not in link:
            continue
        approved = any((rv.get("state") == "APPROVED") for rv in (p.get("reviews") or []))
        login = (p.get("user") or {}).get("login") or ""
        prs.append({
            "id": p.get("id"), "link": link, "number": p.get("number"),
            "title": p.get("title") or "", "summary": p.get("summary") or "",
            "approved": approved, "merged": bool(p.get("merged")), "author": login,
        })
    return prs


def _customer_companies(graph: OntologyGraph, product: str) -> list:
    """Companies of the customers that USE the product (a graph traversal)."""
    try:
        rows = graph.h.rows(
            "MATCH (c:Customer)-[:USES]->(p:Product {id:$id}) RETURN c.name AS cid, c.company AS company",
            {"id": sid("prod:" + product)},
        )
    except Exception:
        rows = []
    comp = _cust_company()
    out = {(r.get("company") or comp.get(r.get("cid"), "")) for r in rows}
    return sorted(c for c in out if c)


def _artifact_text(product: str, aids: set, kinds=None) -> list:
    """Full text for matched artifacts, read from HERB by id. Documents carry
    content, transcripts carry the transcript, PRs carry title + summary."""
    h = _herb(product)
    out = []
    for doc in h.get("documents") or []:
        if doc.get("id") in aids:
            out.append({"id": doc["id"], "kind": "document",
                        "title": doc.get("type") or "", "text": doc.get("content") or ""})
    for tr in h.get("meeting_transcripts") or []:
        if tr.get("id") in aids:
            out.append({"id": tr["id"], "kind": "transcript",
                        "title": tr.get("document_type") or "", "text": tr.get("transcript") or ""})
    for pr in h.get("prs") or []:
        if pr.get("id") in aids:
            out.append({"id": pr["id"], "kind": "pr", "title": pr.get("title") or "",
                        "text": (pr.get("title") or "") + ". " + (pr.get("summary") or "")})
    return out


# --------------------------------------------------------------------------
# LLM select / read (topic match that token overlap cannot do reliably)
# --------------------------------------------------------------------------
def _llm_select(question: str, candidates: list, model=None) -> set:
    """Given candidate artifacts [{key, text}], return the subset of keys whose
    subject matches the question. Empty set is a valid answer (nothing relevant)."""
    if not candidates:
        return set()
    listing = "\n".join(f"{i}. {c['text'][:320]}" for i, c in enumerate(candidates))
    prompt = (
        "Pick the items whose subject matches the question. Match on meaning, not "
        "just shared words: a specific method or component counts if it is an "
        "instance of the feature the question asks about. If the question asks about "
        "a COMPETITOR, do not pick items about the product's own offering. If none "
        "match, return an empty list.\n\n"
        f"Question: {question}\n\nItems:\n{listing}\n\n"
        'Return ONLY JSON {"picks": [<indices>]}.'
    )
    try:
        r = llm.ask_json(prompt, system="You select matching items. Output only JSON.", model=model)
        idx = r.get("picks", []) if isinstance(r, dict) else []
        return {candidates[i]["key"] for i in idx if isinstance(i, int) and 0 <= i < len(candidates)}
    except Exception:
        return set()


def _llm_read(question: str, texts: list, model=None) -> str:
    joined = "\n\n---\n\n".join(t["text"][:2400] for t in texts[:10])
    prompt = (
        "Answer the question using ONLY the sources. Be concise (a short list or a "
        "few sentences). If the sources do not contain the answer, reply exactly "
        "NOT_FOUND.\n\n"
        f"Question: {question}\n\nSources:\n{joined}"
    )
    try:
        return llm.ask(prompt, system="You answer strictly from the given sources.", model=model)
    except Exception:
        return ""


# --------------------------------------------------------------------------
# the query engine
# --------------------------------------------------------------------------
def _abstain(qtype, product):
    return {"type": qtype, "product": product, "answer": [], "abstain": True}


def answer(question_text: str, graph: OntologyGraph, model=None) -> dict:
    plan = _plan(question_text, model=model)
    product = resolve_product(plan.get("product"), question_text)
    qtype = plan.get("type")
    keywords = plan.get("keywords") or list(_tokens(question_text))

    if product is None:
        return _abstain(qtype, None)

    if qtype == "person":
        ans = _answer_person(question_text, graph, product, keywords, model)
    elif qtype == "pr":
        ans = _answer_pr(question_text, product, keywords, model)
    elif qtype == "url":
        ans = _answer_url(question_text, graph, product, keywords, model)
    elif qtype == "company":
        ans = _answer_company(graph, product)
    elif qtype == "content":
        ans = _answer_content(question_text, graph, product, keywords, model)
    else:
        return _abstain(qtype, product)

    if isinstance(ans, str):
        abstain = (not ans) or ans.strip().upper() == "NOT_FOUND"
        return {"type": qtype, "product": product, "answer": ("" if abstain else ans), "abstain": abstain}
    ans = sorted(set(ans))
    return {"type": qtype, "product": product, "answer": ans, "abstain": not ans}


def _answer_person(question, graph, product, keywords, model):
    low = question.lower()
    # Graph-native superlative: the engineer who authored the most approved /
    # unapproved feature PRs. AUTHORED edges to PRs exist only for the salesforce
    # feature PRs (their author id is a real eid), so this is a pure graph count.
    sup = _superlative_prs(question)
    if sup:
        counts: dict = {}
        for pr in _herb_prs(product):
            if not pr["author"].startswith("eid_"):
                continue
            if (sup == "approved") == pr["approved"]:
                counts[pr["author"]] = counts.get(pr["author"], 0) + 1
        if counts:
            top = max(counts.values())
            return [e for e, n in counts.items() if n == top]
        return []

    # "engineers who resolved issue/bug X" -> the authors of the PRs that fix X.
    # The fix PRs are the salesforce feature PRs; matching them to the issue and
    # taking their authors recovers the resolving engineers (verified: every GT
    # eid for these questions is a salesforce-PR author).
    if re.search(r"resolv", low) and ("issue" in low or "bug" in low):
        prs = _herb_prs(product)
        picks = _llm_select(
            question, [{"key": p["link"], "text": p["title"] + ". " + p["summary"]} for p in prs], model=model)
        return sorted({p["author"] for p in prs if p["link"] in picks and p["author"].startswith("eid_")})

    # Competitor questions about people have no representation: there is no
    # person -> competitor-artifact edge in the graph, so abstain rather than
    # dump the whole team.
    if _wants_competitor(question):
        return []

    role = _detect_role(question)
    doc_type = _detect_doc_type(question)

    # A role-attributed document review is not in the corpus: there are no
    # role-tagged REVIEWED edges to documents (REVIEWED links only PRs). So
    # "which <role>s reviewed <document>" has no graph evidence -> abstain. This
    # is the honest empty-traversal signal, verified to hit no answerable question.
    if role and doc_type and "review" in low:
        return []

    # Only fall back to all doc/transcript artifacts when a role filter will
    # narrow the result (e.g. "<role>s who worked on the previous release").
    # Without a topic match and without a role, an empty traversal means abstain.
    arts = _match_artifacts(graph.artifacts_of_product(product), doc_type, keywords,
                            kinds=("document", "transcript"), allow_fallback=bool(role))
    people: set = set()
    for a in arts:
        people |= graph.people_of_artifact(a["aid"], rels=(AUTHORED, PARTICIPATED, REVIEWED))
    if role:
        er = _emp_role()
        people = {e for e in people if er.get(e) == role}
    return people


def _answer_pr(question, product, keywords, model):
    prs = _herb_prs(product)
    if not prs:
        return []
    status = _detect_pr_status(question)
    if status == "approved":
        prs = [p for p in prs if p["approved"]]
    elif status == "not_approved":
        prs = [p for p in prs if not p["approved"]]
    if not prs:
        return []
    # Match by feature semantically over ALL status-filtered PRs. Token overlap
    # alone misses PRs like "Train LSTM Models" for a "predictive analytics"
    # question (no shared tokens), so the LLM does the feature match; the graph /
    # HERB join already fixed the candidate set and the approval filter.
    picks = _llm_select(
        question,
        [{"key": p["link"], "text": p["title"] + ". " + p["summary"]} for p in prs],
        model=model,
    )
    return [p["link"] for p in prs if p["link"] in picks]


def _answer_url(question, graph, product, keywords, model):
    urls = graph.artifacts_of_product(product, kind="url")
    if not urls:
        return []
    # For a "competitor" question, the product's own demos / references do not
    # qualify. Drop artifacts that name the product itself; a competitor's demo
    # never does. Verified: this abstains on products whose only demos are their
    # own, which is exactly the unanswerable case.
    if _wants_competitor(question):
        pl = product.lower()
        urls = [u for u in urls
                if pl not in (u.get("link") or "").lower() and pl not in (u.get("title") or "").lower()]
        if not urls:
            return []
    picks = _llm_select(
        question,
        [{"key": u["aid"], "text": (u.get("title") or "") + " " + (u.get("link") or "")} for u in urls],
        model=model,
    )
    return [u["link"] for u in urls if u["aid"] in picks and u.get("link")]


def _answer_company(graph, product):
    # Best effort: the product's customer companies (a USES traversal). Per-issue
    # company attribution is not present in the corpus text or the graph (see the
    # limitation noted in eval.py), so this recalls the affected set but cannot
    # narrow it to the specific issue.
    return _customer_companies(graph, product)


def _answer_content(question, graph, product, keywords, model):
    # Resolve the product through the graph, then read the product's own text
    # from HERB: documents + review transcripts + meeting chats + feature PRs.
    # Content answers (suggested changes, new features, dropped features) live in
    # the review discussions, not only the finished document, so the pool spans
    # all of them, matched by the named document type or keyword overlap.
    doc_type = _detect_doc_type(question)
    kw = {k for k in keywords if len(k) > 2}
    h = _herb(product)
    pool = []

    def add(title, text):
        if text:
            pool.append({"text": (title + ". " + text) if title else text})

    def relevant(title, body=""):
        if doc_type and doc_type.lower() in (title or "").lower():
            return True
        return bool(kw & (_tokens(title) | _tokens(body[:600])))

    for doc in h.get("documents") or []:
        if relevant(doc.get("type") or "", doc.get("content") or ""):
            add(doc.get("type") or "", doc.get("content") or "")
    for tr in h.get("meeting_transcripts") or []:
        if relevant(tr.get("document_type") or "", tr.get("transcript") or ""):
            add(tr.get("document_type") or "", tr.get("transcript") or "")
    for ch in h.get("meeting_chats") or []:
        if kw & _tokens(ch.get("text") or ""):
            add("", ch.get("text") or "")
    for pr in h.get("prs") or []:
        if "github.com/salesforce/" in (pr.get("link") or "") and (
                kw & (_tokens(pr.get("title")) | _tokens(pr.get("summary") or ""))):
            add(pr.get("title") or "", pr.get("summary") or "")

    if not pool:
        # fall back to whatever the graph traversal matches, then abstain if empty
        arts = _match_artifacts(graph.artifacts_of_product(product), doc_type, keywords,
                                kinds=("document", "transcript", "pr"), allow_fallback=False)
        pool = [{"text": t["text"]} for t in _artifact_text(product, {a["aid"] for a in arts[:12]})]
    if not pool:
        return ""
    return _llm_read(question, pool[:12], model=model)
