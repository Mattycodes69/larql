# larql-python

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

PyO3 bindings packaged as `larql._native`, with Python convenience modules
for graph operations, direct vindex arrays, LQL sessions, walks and traces.
This is a native extension, not a subprocess wrapper around the CLI.

## Build from this checkout

```bash
cd crates/larql-python
uv sync --no-install-project --group dev
uv run --no-sync maturin develop --release
uv run --no-sync pytest tests/
```

The repository toolchain applies to the Rust extension. Use this source-build
workflow when checking APIs against the current checkout; package metadata does
not prove a matching wheel has been published.

## What is exposed

[python/larql/__init__.py](python/larql/__init__.py) exports the supported Python
surface; [src/lib.rs](src/lib.rs) registers native types/functions. Direct
`Vindex` access is implemented in [src/vindex.rs](src/vindex.rs), LQL sessions
in [src/session.rs](src/session.rs), walks and traces in their respective modules.

```python
import larql

graph = larql.Graph()
graph.add_edge(larql.Edge("France", "capital", "Paris"))
```

`larql.session(path)` binds V2 or VINDEX3 through LQL. For example:

```python
session = larql.session("model.vindex3")
print(session.query_text("STATS"))
print(session.query_text('INFER "Hello" GENERATE 16'))
```

Direct `larql.load(...)` and `session.vindex` expose V2 `VectorIndex` arrays.
The session loads that view lazily; accessing it on V3 raises
`NotImplementedError` with guidance to use `query()`. A successful `USE` changes
which artifact the session refers to and invalidates its cached array view.
The full `Vindex3Runtime`, observation and intervention-record APIs remain
available through Rust/CLI rather than separate Python wrappers.

## Threads

Loads, forward passes, generation, trace capture/projection, KNN, WALK,
DESCRIBE, INSERT and LQL statements release the GIL while they run, so
other Python threads keep going. `Vindex`, `WalkModel`, `ResidualTrace` and
`Session` can be shared between threads: a `Vindex`'s mutations (`insert`,
`delete`, `set_*`) take a write lock and are atomic with respect to
concurrent reads, concurrent `infer()` calls on one `Vindex` run one at a
time, and one `Session` executes one statement at a time. `Graph` and its
algorithms still hold the GIL. [tests/test_threading.py](tests/test_threading.py)
measures the release.

See the [Python interface guide](../../docs/larql-python.md),
[tests](tests/) and [runtime surface map](../../docs/runtime-surfaces.md).
Optional MLX/streaming helpers retain their own model and dependency requirements.
