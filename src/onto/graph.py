"""Enterprise ontology on HydraDB: employees, customers, products, orgs and
artifacts (documents, PRs, transcripts, urls), with typed edges between them.

Deep-search queries (who authored/reviewed the report on X, which PRs for feature
Y were not approved, which company reported the most issues) become graph
traversals and entity resolution, not vector similarity. Unanswerable queries are
detected the same way Track 3 does it: an empty bounded traversal means the answer
is not in the corpus, so the system abstains instead of inventing one.
"""
from __future__ import annotations
import hashlib
from .hydra_client import Hydra

EMP, CUST, PROD, ORG, ART = "Employee", "Customer", "Product", "Org", "Artifact"
ON_TEAM, USES, AUTHORED, REVIEWED, PARTICIPATED, REPORTS_TO, ABOUT = (
    "ON_TEAM", "USES", "AUTHORED", "REVIEWED", "PARTICIPATED", "REPORTS_TO", "ABOUT")


def sid(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=7).digest(), "big")


def _clip(s, n=400):
    s = str(s or "")
    return s[:n]


class OntologyGraph:
    def __init__(self, hydra: Hydra | None = None):
        self.h = hydra or Hydra()

    def _edges(self, rows, slabel, dlabel, rel):
        cy = (f"UNWIND $rows AS row MATCH (s:{slabel} {{id: row.s}}), (d:{dlabel} {{id: row.d}}) "
              f"CREATE (s)-[:{rel} {{id: row.eid}}]->(d)")
        for i in range(0, len(rows), 1000):
            self.h.query(cy, {"rows": rows[i:i + 1000]})

    def load(self, o: dict):
        emp = [{"vertex": sid("emp:" + e["eid"]), "name": e["eid"], "ename": e["name"],
                "role": e.get("role", ""), "location": e.get("location", ""), "org": e.get("org", "")}
               for e in o["employees"]]
        self.h.upsert_vertices(emp, label=EMP, props=["name", "ename", "role", "location", "org"])
        cust = [{"vertex": sid("cust:" + c["cid"]), "name": c["cid"], "company": c.get("company", ""),
                 "cname": c.get("name", "")} for c in o["customers"]]
        self.h.upsert_vertices(cust, label=CUST, props=["name", "company", "cname"])
        prod = [{"vertex": sid("prod:" + p["name"]), "name": p["name"]} for p in o["products"]]
        self.h.upsert_vertices(prod, label=PROD, props=["name"])
        org = [{"vertex": sid("org:" + g["name"]), "name": g["name"]} for g in o["orgs"]]
        self.h.upsert_vertices(org, label=ORG, props=["name"])
        art = [{"vertex": sid("art:" + a["aid"]), "name": a["aid"], "kind": a.get("kind", ""),
                "title": _clip(a.get("title")), "product": a.get("product", ""),
                "link": _clip(a.get("link"), 300)} for a in o["artifacts"]]
        self.h.upsert_vertices(art, label=ART, props=["name", "kind", "title", "product", "link"])

        E = o["edges"]
        self._edges([{"s": sid("emp:" + a), "d": sid("prod:" + b), "eid": sid(f"{a}|OT|{b}")}
                     for a, b in E["on_team"]], EMP, PROD, ON_TEAM)
        self._edges([{"s": sid("cust:" + a), "d": sid("prod:" + b), "eid": sid(f"{a}|US|{b}")}
                     for a, b in E["uses"]], CUST, PROD, USES)
        self._edges([{"s": sid("emp:" + a), "d": sid("art:" + b), "eid": sid(f"{a}|AU|{b}")}
                     for a, b in E["authored"]], EMP, ART, AUTHORED)
        self._edges([{"s": sid("emp:" + a), "d": sid("art:" + b), "eid": sid(f"{a}|RV|{b}")}
                     for a, b in E["reviewed"]], EMP, ART, REVIEWED)
        self._edges([{"s": sid("emp:" + a), "d": sid("art:" + b), "eid": sid(f"{a}|PA|{b}")}
                     for a, b in E["participated"]], EMP, ART, PARTICIPATED)
        self._edges([{"s": sid("emp:" + a), "d": sid("emp:" + b), "eid": sid(f"{a}|RT|{b}")}
                     for a, b in E["reports_to"]], EMP, EMP, REPORTS_TO)
        self._edges([{"s": sid("art:" + a), "d": sid("prod:" + b), "eid": sid(f"{a}|AB|{b}")}
                     for a, b in E["about"]], ART, PROD, ABOUT)
        return {k: (len(v) if isinstance(v, list) else v) for k, v in
                {**{n: o[n] for n in ("employees", "customers", "products", "orgs", "artifacts")},
                 **{f"edge_{k}": v for k, v in E.items()}}.items()}

    # ---------- graph primitives for the query engine ----------
    def artifacts_of_product(self, product, kind=None):
        rs = self.h.rows("MATCH (a:Artifact)-[:ABOUT]->(p:Product {id:$id}) "
                         "RETURN a.name AS aid, a.kind AS kind, a.title AS title, a.link AS link",
                         {"id": sid("prod:" + product)})
        return [r for r in rs if not kind or r.get("kind") == kind]

    def people_of_artifact(self, aid, rels=(AUTHORED, PARTICIPATED, REVIEWED)):
        out = set()
        for rel in rels:
            for r in self.h.rows(f"MATCH (e:Employee)-[:{rel}]->(a:Artifact {{id:$id}}) "
                                 "RETURN e.name AS eid", {"id": sid("art:" + aid)}):
                out.add(r["eid"])
        return out

    def team_of_product(self, product, role=None):
        rs = self.h.rows("MATCH (e:Employee)-[:ON_TEAM]->(p:Product {id:$id}) "
                         "RETURN e.name AS eid, e.role AS role, e.ename AS name",
                         {"id": sid("prod:" + product)})
        return [r for r in rs if not role or (r.get("role", "").lower() == role.lower())]

    def count(self, label):
        return self.h.scalar(f"MATCH (n:{label}) RETURN count(*) AS c") or 0
