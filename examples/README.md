# ARIA example gallery

Three self-contained scripts showcasing ARIA's distinctive capabilities.
Each script uses its own `.aria-example-*` state directory, so they do not
stomp on each other or on the main REPL's `.aria/`.

| Example | What it shows |
| --- | --- |
| [`coding_agent.py`](./coding_agent.py) | Program synthesis + library reuse: ARIA synthesizes a Python function for a computation, validates it in a sandbox, and promotes it into a persistent library. Repeat queries of the same shape hit the cached primitive. |
| [`research_agent.py`](./research_agent.py) | Semantic memory ingestion + search: 10 research notes are embedded with the stdlib hash-embedder and retrieved by cosine similarity across five clustered queries. |
| [`personal_org.py`](./personal_org.py) | World model + remember/recall + curiosity loop: seed user identity and entities, record facts, force an idle curiosity tick that self-generates follow-up intents, and query the self-model. |

## Running

No install required — each script prepends `src/` to `sys.path` and uses
only the stdlib-backed stubs (no API keys needed).

```sh
python examples/coding_agent.py
python examples/research_agent.py
python examples/personal_org.py
```

Python 3.11+ is required (same as the package).
