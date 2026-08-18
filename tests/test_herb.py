"""Tests for the HERB ontology parser. No network, no LLM."""

import json
from pathlib import Path

import pytest

from onto.herb import build

REPO = Path(__file__).resolve().parents[1]
HERB = REPO / "data" / "HERB"


@pytest.fixture(scope="module")
def onto(tmp_path_factory):
    out = tmp_path_factory.mktemp("onto") / "ontology.json"
    return build(herb_dir=str(HERB), out_path=str(out))


def test_nodes_present(onto):
    assert len(onto["employees"]) >= 500
    assert len(onto["products"]) == 30
    assert len(onto["customers"]) > 0
    assert len(onto["orgs"]) > 0
    assert len(onto["artifacts"]) > 0
    # every artifact carries the required fields
    kinds = {"document", "pr", "transcript", "url"}
    for art in onto["artifacts"]:
        assert set(art) >= {"aid", "kind", "title", "product", "link"}
        assert art["kind"] in kinds


def test_all_edge_lists_non_empty(onto):
    edges = onto["edges"]
    expected = {"on_team", "uses", "authored", "reviewed", "participated", "reports_to", "about"}
    assert set(edges) == expected
    for name, pairs in edges.items():
        assert len(pairs) > 0, f"edge {name} is empty"
        for pair in pairs:
            assert len(pair) == 2


def test_edges_reference_real_nodes(onto):
    eids = {e["eid"] for e in onto["employees"]}
    cids = {c["cid"] for c in onto["customers"]}
    prods = {p["name"] for p in onto["products"]}
    aids = {a["aid"] for a in onto["artifacts"]}
    for eid, prod in onto["edges"]["on_team"]:
        assert eid in eids and prod in prods
    for cid, prod in onto["edges"]["uses"]:
        assert cid in cids and prod in prods
    for eid, aid in onto["edges"]["authored"]:
        assert eid in eids and aid in aids
    for eid, aid in onto["edges"]["reviewed"]:
        assert eid in eids and aid in aids
    for eid, aid in onto["edges"]["participated"]:
        assert eid in eids and aid in aids
    for child, parent in onto["edges"]["reports_to"]:
        assert child in eids and parent in eids and child != parent
    for aid, prod in onto["edges"]["about"]:
        assert aid in aids and prod in prods


def test_questions_answerable_and_unanswerable(onto):
    qs = onto["questions"]
    answerable = [q for q in qs if q["answerable"]]
    unanswerable = [q for q in qs if not q["answerable"]]
    assert answerable, "no answerable questions"
    assert unanswerable, "no unanswerable questions"
    # every product contributes at least one question
    assert {q["product"] for q in qs} == {p["name"] for p in onto["products"]}


def test_person_answerable_have_ground_truth(onto):
    person_q = [q for q in onto["questions"] if q["answerable"] and q["type"] == "person"]
    assert person_q, "expected person-type answerable questions"
    for q in person_q:
        assert q["ground_truth"], f"empty ground_truth: {q['question']!r}"


def test_spot_check_team_eids_exist(onto):
    """At least one product's team eids all resolve to real employees."""
    eids = {e["eid"] for e in onto["employees"]}
    team_by_product = {}
    for eid, prod in onto["edges"]["on_team"]:
        team_by_product.setdefault(prod, []).append(eid)
    ok = [p for p, team in team_by_product.items() if team and all(e in eids for e in team)]
    assert ok, "no product has a fully-resolved team"


def test_deterministic_idempotent(tmp_path):
    out = tmp_path / "ontology.json"
    build(herb_dir=str(HERB), out_path=str(out))
    first = out.read_text()
    build(herb_dir=str(HERB), out_path=str(out))
    second = out.read_text()
    assert first == second
    json.loads(first)  # valid JSON
