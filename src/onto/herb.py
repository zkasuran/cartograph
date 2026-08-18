"""Parse the HERB enterprise dataset into a normalized ontology.

Pure parsing: no database, no LLM, no network. ``build`` reads the HERB
metadata + per-product files and emits nodes, edges and the question set as a
single deterministic JSON document.

Schema surprises handled here (see module tests):
- ``unanswerable_questions`` entries are plain strings, not dicts, so they
  carry no type / ground_truth / citations.
- PR ``user.login`` and review authors use an ``EMP_`` namespace that is
  absent from ``employee.json`` (which is keyed by ``eid_``). Edges are only
  emitted to employees that exist as nodes; unresolved ids are counted and
  reported via the returned ``_stats`` block rather than left dangling.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# Groups nested directly under a salesforce_team root (the VP of Engineering).
_ROOT_GROUPS = (
    "engineering_leads",
    "product_managers",
    "tech_architects",
    "ux_researchers",
    "marketing_research_analysts",
    "chief_product_officers",
    "marketing_managers",
)


def _load(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _title(obj: dict) -> str | None:
    for key in ("title", "summary", "description", "document_type", "type"):
        val = obj.get(key)
        if val:
            return val
    return None


def _link(obj: dict) -> str | None:
    for key in ("link", "document_link"):
        val = obj.get(key)
        if val:
            return val
    return None


def build(herb_dir: str = "data/HERB", out_path: str = "data/ontology.json") -> dict:
    """Build the normalized ontology and write it to ``out_path``.

    Returns the ontology dict. Deterministic and idempotent: re-running on the
    same inputs writes byte-identical output.
    """
    herb = Path(herb_dir)
    emp_raw = _load(herb / "metadata" / "employee.json")
    cust_raw = _load(herb / "metadata" / "customers_data.json")
    sf_teams = _load(herb / "metadata" / "salesforce_team.json")
    product_files = sorted((herb / "products").glob("*.json"))

    emp_ids = set(emp_raw)
    name_to_eids: dict[str, set] = defaultdict(set)
    for eid, rec in emp_raw.items():
        name_to_eids[rec.get("name")].add(eid)

    def resolve_person(token: Any) -> str | None:
        """Resolve a participant token to a known eid, or None if unresolvable."""
        if not isinstance(token, str):
            return None
        if token in emp_ids:
            return token
        candidates = name_to_eids.get(token)
        if candidates and len(candidates) == 1:  # only unambiguous names
            return next(iter(candidates))
        return None

    employees = [
        {
            "eid": eid,
            "name": emp_raw[eid].get("name"),
            "role": emp_raw[eid].get("role"),
            "location": emp_raw[eid].get("location"),
            "org": emp_raw[eid].get("org"),
        }
        for eid in sorted(emp_raw)
    ]
    customers = [
        {
            "cid": c["id"],
            "name": c.get("name"),
            "company": c.get("company"),
            "role": c.get("role"),
        }
        for c in sorted(cust_raw, key=lambda c: c["id"])
    ]
    orgs = [{"name": o} for o in sorted({r.get("org") for r in emp_raw.values() if r.get("org")})]

    # reports_to: child -> parent, straight from the salesforce_team nesting.
    reports_to: set = set()
    for tree in sf_teams:
        root = tree.get("employee_id")
        for group in _ROOT_GROUPS:
            for member in tree.get(group) or []:
                mid = member.get("employee_id")
                if mid and root and mid != root and mid in emp_ids and root in emp_ids:
                    reports_to.add((mid, root))
                if group == "engineering_leads":
                    for sub in (member.get("engineers") or []) + (member.get("qa_specialists") or []):
                        sid = sub.get("employee_id")
                        if sid and mid and sid != mid and sid in emp_ids and mid in emp_ids:
                            reports_to.add((sid, mid))

    products: list = []
    artifacts: dict = {}  # aid -> node; first occurrence (sorted product order) wins
    on_team: set = set()
    uses: set = set()
    authored: set = set()
    reviewed: set = set()
    participated: set = set()
    about: set = set()
    questions: list = []

    stats = {"slack_skipped": 0, "author_refs_dropped": 0, "review_refs_dropped": 0, "participant_refs_dropped": 0}

    def register(aid, kind, obj, product) -> None:
        if aid and aid not in artifacts:
            artifacts[aid] = {"aid": aid, "kind": kind, "title": _title(obj), "product": product, "link": _link(obj)}

    for pf in product_files:
        product = pf.stem
        products.append({"name": product})
        data = _load(pf)
        stats["slack_skipped"] += len(data.get("slack") or [])

        for eid in data.get("team") or []:
            if eid in emp_ids:
                on_team.add((eid, product))
        for cid in data.get("customers") or []:
            uses.add((cid, product))

        for doc in data.get("documents") or []:
            aid = doc.get("id")
            if not aid:
                continue
            register(aid, "document", doc, product)
            about.add((aid, product))
            author = doc.get("author")
            if author in emp_ids:
                authored.add((author, aid))
            elif author:
                stats["author_refs_dropped"] += 1

        for pr in data.get("prs") or []:
            aid = pr.get("id")
            if not aid:
                continue
            register(aid, "pr", pr, product)
            about.add((aid, product))
            login = (pr.get("user") or {}).get("login")
            if login in emp_ids:
                authored.add((login, aid))
            elif login:
                stats["author_refs_dropped"] += 1
            for review in pr.get("reviews") or []:
                rlogin = (review.get("user") or {}).get("login")
                if rlogin in emp_ids:
                    reviewed.add((rlogin, aid))
                elif rlogin:
                    stats["review_refs_dropped"] += 1

        for tr in data.get("meeting_transcripts") or []:
            aid = tr.get("id")
            if not aid:
                continue
            register(aid, "transcript", tr, product)
            about.add((aid, product))
            for token in tr.get("participants") or []:
                person = resolve_person(token)
                if person:
                    participated.add((person, aid))
                else:
                    stats["participant_refs_dropped"] += 1

        for url in data.get("urls") or []:
            aid = url.get("id")
            if not aid:
                continue
            register(aid, "url", url, product)
            about.add((aid, product))

        for q in data.get("answerable_questions") or []:
            questions.append({
                "product": product,
                "question": q.get("question"),
                "type": q.get("type"),
                "ground_truth": q.get("ground_truth") or [],
                "citations": q.get("citations") or [],
                "answerable": True,
            })
        for q in data.get("unanswerable_questions") or []:
            if isinstance(q, str):
                questions.append({
                    "product": product,
                    "question": q,
                    "type": None,
                    "ground_truth": [],
                    "citations": [],
                    "answerable": False,
                })
            else:
                questions.append({
                    "product": product,
                    "question": q.get("question"),
                    "type": q.get("type"),
                    "ground_truth": q.get("ground_truth") or [],
                    "citations": q.get("citations") or [],
                    "answerable": False,
                })

    def pairs(edge_set: set) -> list:
        return [list(pair) for pair in sorted(edge_set)]

    ontology = {
        "employees": employees,
        "customers": customers,
        "products": products,
        "orgs": orgs,
        "artifacts": [artifacts[aid] for aid in sorted(artifacts)],
        "edges": {
            "on_team": pairs(on_team),
            "uses": pairs(uses),
            "authored": pairs(authored),
            "reviewed": pairs(reviewed),
            "participated": pairs(participated),
            "reports_to": pairs(reports_to),
            "about": pairs(about),
        },
        "questions": questions,
        "_stats": stats,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(ontology, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return ontology


if __name__ == "__main__":
    o = build()
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in o.items() if k != "edges"}, indent=2))
    print({k: len(v) for k, v in o["edges"].items()})
