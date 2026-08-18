"""Loader integrity and graph traversal primitives against a live HydraDB node.

These pin the load-bearing graph layer the query engine sits on: the loader puts
every ontology node into HydraDB, the ABOUT / ON_TEAM / AUTHORED traversals return
exactly the edges the normalized ontology declares, same-name employees stay
distinct nodes (the entity-resolution claim), and REPORTS_TO walks the real
hierarchy. Read-only against the already-loaded node; a conditional load keeps the
file self-sufficient on a cold node without duplicating edges on a warm one. Tests
skip cleanly when no HydraDB node is reachable.
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

from onto.graph import ART, CUST, EMP, ORG, PROD, OntologyGraph, sid  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ONTO = json.loads((REPO / "data" / "ontology.json").read_text(encoding="utf-8"))

_EMP_IDS = {e["eid"] for e in ONTO["employees"]}
_ARTS = {a["aid"]: a for a in ONTO["artifacts"]}
_LABEL_KEY = {EMP: "employees", CUST: "customers", PROD: "products", ART: "artifacts", ORG: "orgs"}
_PRODUCT = "ActionGenie"       # a real product with artifacts, a team and authored docs
_SHARED_NAME = "Charlie Brown"  # a display name held by several distinct employees


def _node_up() -> bool:
    base = os.environ.get("HYDRA_HTTP", "http://127.0.0.1:8443")
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/healthz", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


requires_node = pytest.mark.skipif(not _node_up(), reason="HydraDB node not reachable")


@pytest.fixture(scope="module")
def graph():
    g = OntologyGraph()
    # Node counts are idempotent (vertices MERGE by id), so only load when the node
    # is not already populated. A warm node is left untouched because edges are
    # CREATEd, and a needless reload would duplicate relationships.
    if g.count(EMP) != len(ONTO["employees"]):
        g.load(ONTO)
    return g


def _about_aids(product):
    return {aid for aid, prod in ONTO["edges"]["about"] if prod == product}


def _team_eids(product):
    return {eid for eid, prod in ONTO["edges"]["on_team"] if prod == product}


def _authored_document(product):
    """A document artifact ABOUT the product whose author resolves to a real eid."""
    about = _about_aids(product)
    authored = {}
    for eid, aid in ONTO["edges"]["authored"]:
        authored.setdefault(aid, set()).add(eid)
    for aid in sorted(about):
        if _ARTS[aid]["kind"] == "document" and authored.get(aid):
            real = sorted(authored[aid] & _EMP_IDS)
            if real:
                return aid, real[0]
    raise AssertionError(f"no authored document ABOUT {product}")


@requires_node
def test_loader_counts_match_ontology(graph):
    """Every normalized ontology node reaches HydraDB: per-label node counts equal
    the ontology list lengths (Employee 530, Customer 120, Product 30, Artifact
    4858, Org 6). A dropped or half-loaded label fails here."""
    for label, key in _LABEL_KEY.items():
        assert graph.count(label) == len(ONTO[key]), f"{label} count != len({key})"


@requires_node
def test_traversal_primitives_match_edges(graph):
    """The three query-engine traversals return exactly the edges the ontology
    declares: artifacts_of_product is the ABOUT set, team_of_product is the ON_TEAM
    set (all real employees), and people_of_artifact on a known document returns its
    real author. A fabricated or missing edge fails here."""
    aids = {r["aid"] for r in graph.artifacts_of_product(_PRODUCT)}
    assert aids, "artifacts_of_product returned nothing"
    assert aids == _about_aids(_PRODUCT)
    assert aids <= set(_ARTS)                 # every returned aid is a real artifact

    team = {r["eid"] for r in graph.team_of_product(_PRODUCT)}
    assert team, "team_of_product returned nothing"
    assert team == _team_eids(_PRODUCT)
    assert team <= _EMP_IDS                    # every team eid is a real employee

    doc_aid, author = _authored_document(_PRODUCT)
    people = graph.people_of_artifact(doc_aid)
    assert author in people, f"{author} not among {sorted(people)}"
    assert set(people) <= _EMP_IDS


@requires_node
def test_same_name_employees_are_distinct_nodes(graph):
    """The entity-resolution claim: employees sharing a display name are distinct
    graph nodes with distinct eids, told apart by role/org. 'Charlie Brown' is
    several different people; the graph keeps each as its own Employee node carrying
    its own attributes rather than collapsing them into one."""
    same = [e for e in ONTO["employees"] if e["name"] == _SHARED_NAME]
    assert len(same) >= 2, f"expected 2+ employees named {_SHARED_NAME!r}"
    eids = [e["eid"] for e in same]
    assert len(set(eids)) == len(eids), "shared-name employees must have distinct eids"
    keys = {(e.get("role"), e.get("org")) for e in same}
    assert len(keys) >= 2, "shared-name employees are not separable by role/org"
    for e in same:
        rows = graph.h.rows(
            "MATCH (n:Employee {id:$id}) RETURN n.ename AS name, n.role AS role, n.org AS org",
            {"id": sid("emp:" + e["eid"])})
        assert len(rows) == 1, f"{e['eid']} is not a single Employee node"
        assert rows[0]["name"] == _SHARED_NAME
        assert rows[0]["role"] == (e.get("role") or "")
        assert rows[0]["org"] == (e.get("org") or "")


@requires_node
def test_reports_to_hierarchy(graph):
    """A REPORTS_TO edge from the ontology is walkable in the graph: matching the
    child Employee and following REPORTS_TO returns the expected manager eid. A
    broken or mis-directed edge fails here."""
    child, manager = ONTO["edges"]["reports_to"][0]
    assert child in _EMP_IDS and manager in _EMP_IDS
    rows = graph.h.rows(
        "MATCH (e:Employee {id:$id})-[:REPORTS_TO]->(m:Employee) RETURN m.name AS eid",
        {"id": sid("emp:" + child)})
    managers = {r["eid"] for r in rows}
    assert manager in managers, f"expected {manager} in {sorted(managers)}"
