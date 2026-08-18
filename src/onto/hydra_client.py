"""Minimal HydraDB HTTP client, verified against ghcr.io/hydra-db/hydradb:latest.

HydraDB speaks a deliberate OpenCypher subset (see HYDRADB-NOTES.md). This wraps
the JSON query API plus the two-phase bulk-load the engine requires: MERGE
vertices by integer id, then MATCH+CREATE typed edges. Node ids are integers, so
callers map their string keys to ids and keep the mapping.
"""
from __future__ import annotations
import json, os, urllib.request, urllib.error
from typing import Any, Sequence


class HydraError(RuntimeError):
    pass


def cypher_lit(v: Any) -> str:
    """Serialise a Python value to a Cypher literal (for procedure config maps,
    where list params are not accepted and must be inline)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return "'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(cypher_lit(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {cypher_lit(x)}" for k, x in v.items()) + "}"
    raise TypeError(f"no cypher literal for {type(v)}")


class Hydra:
    def __init__(self, base=None, token=None, graph="default", cell="cell-0", namespace="default"):
        self.base = (base or os.environ.get("HYDRA_HTTP", "http://127.0.0.1:8443")).rstrip("/")
        self.token = token or os.environ.get("HYDRA_TOKEN", "local-development-token-32-bytes")
        self.graph, self.cell, self.namespace = graph, cell, namespace

    def query(self, cypher: str, parameters: dict | None = None, consistency: str = "causal") -> dict:
        body: dict[str, Any] = {"cell_id": self.cell, "query": cypher}
        if parameters is not None:
            body["parameters"] = parameters
        if consistency:
            body["consistency"] = consistency
        req = urllib.request.Request(
            f"{self.base}/v1/graphs/{self.graph}/query",
            data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.token}",
                     "X-Graph-Namespace": self.namespace,
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise HydraError(f"HTTP {e.code}: {e.read().decode()[:400]}") from None
        if isinstance(out, dict) and out.get("error"):
            raise HydraError(out["error"].get("message", str(out["error"])))
        return out

    @staticmethod
    def _val(cell):
        return cell.get("value") if isinstance(cell, dict) else cell

    def rows(self, cypher: str, parameters: dict | None = None, **kw) -> list[dict]:
        out = self.query(cypher, parameters, **kw)
        cols = out.get("columns", [])
        return [{c: self._val(v) for c, v in zip(cols, row)} for row in out.get("rows", [])]

    def scalar(self, cypher: str, parameters: dict | None = None, **kw):
        rs = self.rows(cypher, parameters, **kw)
        return next(iter(rs[0].values())) if rs else None

    # ---- two-phase bulk load ----
    def upsert_vertices(self, rows: Sequence[dict], label: str, props: Sequence[str] = (), batch: int = 500):
        """rows: dicts with 'vertex' (int id) + property fields named in `props`."""
        sets = ", ".join([f"n:{label}"] + [f"n.{p} = row.{p}" for p in props])
        cy = f"UNWIND $rows AS row MERGE (n {{id: row.vertex}}) SET {sets}"
        for i in range(0, len(rows), batch):
            self.query(cy, {"rows": list(rows[i:i + batch])})

    def create_edges(self, rows: Sequence[dict], rel: str, label: str, props: Sequence[str] = (), batch: int = 500):
        """rows: dicts with 's','d' (int endpoint ids), 'eid' (int rel id) + props."""
        setp = "".join(f", {p}: row.{p}" for p in props)
        cy = (f"UNWIND $rows AS row MATCH (s:{label} {{id: row.s}}), (d:{label} {{id: row.d}}) "
              f"CREATE (s)-[:{rel} {{id: row.eid{setp}}}]->(d)")
        for i in range(0, len(rows), batch):
            self.query(cy, {"rows": list(rows[i:i + batch])})

    # ---- path procedures (config map inlined; lists cannot be $params) ----
    def _paths(self, proc: str, cfg: dict) -> list[dict]:
        out = self.query(f"CALL algo.{proc}({cypher_lit(cfg)}) YIELD path RETURN path")
        paths = []
        for row in out.get("rows", []):
            cell = row[0]
            paths.append(cell.get("value") if isinstance(cell, dict) else cell)
        return paths

    def sspaths(self, source_id: int, rel_types, direction="outgoing", max_len=4,
                path_count=1000, result_limit=5000):
        # pathCount caps how many paths come back; the default of 1 truncates
        # reachability, so pass a generous bound to enumerate the full frontier.
        cfg = {"sourceNode": source_id, "relTypes": list(rel_types), "relDirection": direction,
               "maxLen": max_len, "pathCount": path_count, "resultLimit": result_limit}
        return self._paths("SSpaths", cfg)

    def reach(self, source_id: int, rel_types, direction="incoming", max_len=6, key="name",
              path_count=2000, result_limit=10000) -> set:
        """Set of node ids reachable from source within max_len hops (default
        incoming = 'what reaches X', i.e. blast radius). Excludes the source."""
        ids = set()
        for p in self.sspaths(source_id, rel_types, direction, max_len, path_count, result_limit):
            for n in p.get("nodes", []):
                if n.get("id") != source_id:
                    ids.add(n.get("id"))
        return ids

    def sppaths(self, source_id: int, target_id: int, rel_types, direction="outgoing", max_len=6, path_count=5):
        cfg = {"sourceNode": source_id, "targetNode": target_id, "relTypes": list(rel_types),
               "relDirection": direction, "maxLen": max_len, "pathCount": path_count}
        return self._paths("SPpaths", cfg)

    def mspaths(self, source_values, target_values, rel_types, label, prop="name",
                direction="outgoing", max_len=4, path_count=5):
        cfg = {"sourceLabel": label, "sourceProperty": prop, "sourceValues": [str(v) for v in source_values],
               "targetLabel": label, "targetProperty": prop, "targetValues": [str(v) for v in target_values],
               "relTypes": list(rel_types), "relDirection": direction, "maxLen": max_len, "pathCount": path_count}
        return self._paths("MSpaths", cfg)

    @staticmethod
    def path_node_names(path: dict, key: str = "name") -> list:
        """Extract a property from each node of a path result, in order."""
        names = []
        for n in path.get("nodes", []):
            p = n.get("properties", {}).get(key)
            names.append(next(iter(p.values())) if isinstance(p, dict) else p)
        return names
