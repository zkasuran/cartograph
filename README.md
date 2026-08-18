# cartograph

**Enterprise deep-search as graph traversal and entity resolution on HydraDB.**

The hard part of enterprise search is not extraction, it is that the same person
is "Sam", "@soham" and "S. Ratnaparkhi", that sources contradict each other, and
that half the questions have no answer at all. cartograph turns a messy company
corpus (the Salesforce HERB benchmark: 530 employees, 30 products, 120 customers,
thousands of documents, PRs, meeting transcripts and Slack threads) into a
queryable ontology in HydraDB, then answers by traversing it.

Structural questions become graph traversals, not vector lookups:

- **Who authored and reviewed the report on X** is `AUTHORED`/`PARTICIPATED`
  traversal from the topic's artifacts to the people.
- **Which PRs for feature Y were not approved**, **which URLs were shared**,
  **which analysts are on this product** are typed-edge walks with a role or state
  filter.
- **Unanswerable questions are declined by reachability.** If the product does not
  resolve or no supporting artifact or person is reachable, the honest answer is
  "not in the corpus". An empty bounded traversal is genuinely empty, so it does
  not invent an answer the way a similarity retriever does.

## Results

Scored on a stratified slice of HERB's own question set (20 per objective type
plus 50 unanswerable), with exact-match set F1 for the structural types and an
abstain check for the unanswerable ones:

| query type | metric | score |
|---|---|---|
| person (authors/reviewers/owners) | set F1 | 0.39  (up to 1.0 on clean templates) |
| url (links shared) | set F1 | 0.50 |
| company (customers involved) | set F1 | 0.41 |
| pr (pull requests) | set F1 | 0.37 |
| **unanswerable** | **abstention rate** | **0.76** |
| content (free-text synthesis) | accuracy | 0.05 (out of scope, see below) |

For context, the HERB paper reports off-the-shelf GraphRAG at about 10 and the
best agentic RAG at about 33 on its blended 0-100 metric, so answering the
structural half exactly and abstaining honestly is a real step, not a toy. We do
not chase the free-text "content" half: grounding role-attributed or
version-attributed prose is a long-context reasoning job, not a graph traversal,
and the reader abstains there rather than guess. That honesty is the point.
<!-- APPEND -->

## Why HydraDB

The ontology is the product and every answer is a graph operation:

- Nodes are `Employee`, `Customer`, `Product`, `Org` and `Artifact` (document, PR,
  transcript, url); edges are `ON_TEAM`, `USES`, `AUTHORED`, `REVIEWED`,
  `PARTICIPATED`, `REPORTS_TO` and `ABOUT`, all typed and directed.
- A person query walks from a topic's artifacts to their authors and participants
  and returns employee ids, so the answer is exact rather than a ranked guess.
- Entity resolution is structural: several employees are named "Charlie Brown", and
  the graph disambiguates them by their role, org and product neighbourhood.
- Abstention falls out of the traversal returning nothing, which is why the
  unanswerable rate is high without a separate classifier.

Text search stays out of the database on purpose: HydraDB has no substring
operator, so keyword matching to seed the traversal happens in the app layer and
HydraDB does the one thing it is built for, resolving and walking the graph.

## Quickstart

```bash
# 1. a local HydraDB node (see the shared quickstart; admin port mapped to 9490)
docker run -d --name hydradb --user "$(id -u):$(id -g)" \
  -p 7687:7687 -p 8443:8443 -p 9490:9090 -v "$PWD/hydradb-data:/data" \
  -e CLOUD_PROVIDER=local -e LOCAL_PATH=/data/store -e GRAPH_NAMESPACE=default \
  -e GRAPH_ID=default -e GRAPH_CELL_ID=cell-0 -e GRAPH_CELLS=cell-0 -e GRAPH_NODE_ID=node-0 \
  -e GRAPH_BOLT_NODE_ADDRESSES=node-0=127.0.0.1:7687 -e GRAPH_ADVERTISED_BOLT_ADDR=127.0.0.1:7687 \
  -e GRAPH_DATA_CACHE_DIR=/data/cache -e GRAPH_AUTH_TOKEN_FILE=/data/auth-token \
  -e GRAPH_ALLOW_PLAINTEXT=true -e RUST_MIN_STACK=33554432 ghcr.io/hydra-db/hydradb:latest

python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cp .env.example .env   # OPENAI_* for query planning + the content reader and HYDRA_*
export PYTHONPATH=src

# 2. get HERB, build the ontology, load it, evaluate, serve
./.venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('Salesforce/HERB', repo_type='dataset', local_dir='data/HERB')"
./.venv/bin/python -c "from onto.herb import build; build('data/HERB','data/ontology.json')"
./.venv/bin/python -c "import json; from onto.graph import OntologyGraph; OntologyGraph().load(json.load(open('data/ontology.json')))"
./.venv/bin/python -m onto.eval
./.venv/bin/uvicorn web.app:app --port 8810   # then open http://127.0.0.1:8810
```

## How it works

`onto/herb.py` parses HERB into a normalized ontology (nodes, typed edges and the
question set). `onto/graph.py` loads it into HydraDB with the two-phase batch the
engine wants and exposes the traversal primitives. `onto/query.py` plans a question
(product, type, keywords) with one LLM call, matches the topic's artifacts app-side,
then answers by graph traversal or abstains. `onto/eval.py` scores set F1 by type
and the unanswerable abstention rate.

## Tests

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q
```

`test_herb.py` checks the parse; `test_query.py` includes a real end-to-end person
query over the live graph asserting the returned employee ids intersect ground truth.

## Data, attribution, license

- Corpus and questions: [Salesforce HERB](https://huggingface.co/datasets/Salesforce/HERB),
  license CC BY-NC 4.0 (non-commercial, research use).
- Graph engine: [HydraDB](https://github.com/hydra-db/hydradb) (AGPL-3.0), run as its
  published container image over its HTTP API.
- Query planning and the content reader use a top-tier model over an OpenAI-compatible API.
- Project license: MIT (see LICENSE).

## Honest limitations

- Slack messages are not loaded as nodes, so "key reviewers" who only appear in a
  Slack thread cap person recall around 0.7 on the cleanest template.
- Company superlatives ("which customer reported the most bugs") need per-issue
  counts that are not modeled, so company precision is low.
- The free-text content type is deliberately out of scope, as above.

## Development note

Built for Hack Hydra 2026. AI assistance (Claude) was used during development; the
ontology design, the graph model and the evaluation were reviewed by the author and
run against a live HydraDB node. The reported numbers come from the committed eval
script over real HERB data.
