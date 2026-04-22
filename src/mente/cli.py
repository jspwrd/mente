"""Unified command-line interface.

All entry points live here so there's exactly one thing to run:

    ./mente              # interactive REPL
    ./mente demo         # scripted demo
    ./mente federated    # hub + peer in one process, real TCP bus between them
    ./mente reset        # wipe .mente/ state
    ./mente test         # smoke tests (bus, synthesis, memory)
    ./mente migrate      # upgrade state files to the current schema
    ./mente --help       # all options
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Any

from .config import MenteConfig
from .logging import get_logger
from .reasoners import Reasoner
from .runtime import Runtime
from .types import Belief, Event, Intent

BANNER = r"""
   _      ___   ___
  /_\    | _ \ |_ _|   /_\
 / _ \   |   /  | |   / _ \
/_/ \_\  |_|_\ |___| /_/ \_\

  persistent, event-driven reasoning process
"""


def _root() -> Path:
    """Project root — so .mente/ lives in the project dir regardless of cwd."""
    return Path(__file__).resolve().parent.parent.parent


def _data_root(name: str = ".mente") -> Path:
    return _root() / name


async def _seed(rt: Runtime) -> None:
    """Seed the world model with a lightweight identity so demos feel grounded."""
    user_name = os.environ.get("USER") or "there"
    await rt.world.assert_belief(
        Belief(entity="user", attribute="name", value=user_name.capitalize())
    )


# ---------------------------------------------------------------------------
# interactive REPL
# ---------------------------------------------------------------------------

async def _run_repl(rt: Runtime) -> None:
    """Read lines from stdin, route through runtime, print responses."""
    print(BANNER)
    print(f"booted. {len(rt.reasoners)} reasoners, {len(rt.tools.list())} tools.")
    print("type a message. /help for commands. /quit to exit.\n")

    loop = asyncio.get_running_loop()
    while True:
        try:
            line = await loop.run_in_executor(None, lambda: input("you> "))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            if line in ("/quit", "/exit", "/q"):
                break
            if line == "/help":
                _print_help()
                continue
            if line == "/state":
                _print_state(rt)
                continue
            if line == "/library":
                _print_library(rt)
                continue
            if line == "/bus":
                _print_bus(rt)
                continue
            if line == "/digest":
                d = rt.consolidator.consolidate()
                for k, v in d.items():
                    print(f"  {k}: {v}")
                continue
            print(f"unknown command: {line} (try /help)")
            continue

        r = await rt.handle_intent(Intent(text=line))
        reasoner = rt.latent.get("last_reasoner")
        print(f"mente[{reasoner}]> {r.text}")


def _print_help() -> None:
    print("""commands:
  /help      this
  /state     current latent state
  /library   synthesized primitives
  /bus       last 20 bus events
  /digest    force a consolidation digest
  /quit      exit
""")


def _print_state(rt: Runtime) -> None:
    for k, v in rt.latent.values.items():
        s = str(v)
        if len(s) > 80:
            s = s[:77] + "..."
        print(f"  {k}: {s}")


def _print_library(rt: Runtime) -> None:
    prims = rt.library.list()
    if not prims:
        print("  (library empty — try 'compute the 10th fibonacci number')")
        return
    for p in prims:
        print(f"  {p.name}  entry={p.entrypoint}  calls={p.invocations}")


def _print_bus(rt: Runtime) -> None:
    for e in rt.bus.recent(n=20):
        payload = str(e.payload)
        if len(payload) > 60:
            payload = payload[:57] + "..."
        print(f"  {e.topic:32s} from={e.origin:20s} {payload}")


# ---------------------------------------------------------------------------
# scripted demo
# ---------------------------------------------------------------------------

DEMO_SCRIPT = [
    "hello",
    "who am I?",
    "what time is it?",
    "remember that the deploy freeze starts Monday",
    "remember that raspberry pis are memory-bandwidth bound",
    "what do you know about hardware?",
    "compute the 15th fibonacci number",
    "what is the factorial of 8",
    "if I gave you three raspberry pis, could you run a 70B model across them?",
    "what are you?",
    "what have you been doing?",
]


async def _run_demo(rt: Runtime) -> None:
    print(BANNER)
    print("scripted demo — watch the router pick reasoners per intent.\n")
    for text in DEMO_SCRIPT:
        r = await rt.handle_intent(Intent(text=text))
        reasoner = rt.latent.get("last_reasoner")
        print(f">>> {text}")
        print(f"    [{reasoner}] {r.text}")
        if text == "what do you know about hardware?":
            # mid-run consolidation so the self-model has something to read
            rt.consolidator.consolidate()

    print("\nfinal library:")
    _print_library(rt)


# ---------------------------------------------------------------------------
# federated demo — hub + peer in one process, real TCP between them
# ---------------------------------------------------------------------------

async def _run_federated(port: int) -> None:
    from .bus import EventBus
    from .discovery import Announcer, Directory, RemoteReasoner, RemoteRequestHandler
    from .specialists import MathSpecialist
    from .tools import ToolRegistry
    from .transport import TCPTransport
    from .world_model import WorldModel

    print(BANNER)
    print(f"federated demo — hub + math peer, TCP bus on port {port}\n")

    # --- hub (main runtime) ---
    hub_id = "hub.main"
    hub_bus = EventBus(transport=TCPTransport(node_id=hub_id, port=port, role="hub"))
    rt = Runtime(root=_data_root(".mente-hub"), node_id=hub_id, bus=hub_bus)
    await rt.start()
    await _seed(rt)
    directory = Directory(bus=rt.bus, self_node_id=hub_id)
    directory.wire()

    # --- peer (math specialist) ---
    peer_id = "peer.math"
    peer_bus = EventBus(transport=TCPTransport(node_id=peer_id, port=port, role="spoke"))
    await peer_bus.start()
    peer_world = WorldModel(bus=peer_bus)
    peer_tools = ToolRegistry()
    peer_reasoners: list[Reasoner] = [MathSpecialist()]
    RemoteRequestHandler(
        bus=peer_bus, node_id=peer_id,
        reasoners=peer_reasoners, world=peer_world, tools=peer_tools,
    ).wire()

    stop = asyncio.Event()
    peer_announcer = Announcer(
        bus=peer_bus, node_id=peer_id,
        reasoners=peer_reasoners, specialization="math",
    )
    hub_announcer = Announcer(bus=rt.bus, node_id=hub_id, reasoners=rt.reasoners)
    tasks = [
        asyncio.create_task(peer_announcer.run(stop)),
        asyncio.create_task(hub_announcer.run(stop)),
    ]

    print("waiting for peer discovery...")
    for _ in range(50):
        if directory.specialists():
            break
        await asyncio.sleep(0.1)

    specialists = directory.specialists()
    print(f"discovered: {[f'{p.node_id}:{p.reasoner}' for p in specialists]}\n")
    for p in specialists:
        remote = RemoteReasoner(bus=rt.bus, node_id=hub_id, target=p)
        # rt.reasoners, rt.router.reasoners, rt.router.metacog.reasoners all
        # reference the same list — append once.
        rt.reasoners.append(remote)

    try:
        await _run_repl(rt)
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await peer_bus.close()
        await rt.shutdown()


# ---------------------------------------------------------------------------
# peer subcommand (for people who want real multi-process)
# ---------------------------------------------------------------------------

async def _run_peer_only(port: int, node_id: str) -> None:
    from .bus import EventBus
    from .discovery import Announcer, RemoteRequestHandler
    from .specialists import MathSpecialist
    from .tools import ToolRegistry
    from .transport import TCPTransport
    from .world_model import WorldModel

    bus = EventBus(transport=TCPTransport(node_id=node_id, port=port, role="spoke"))
    await bus.start()
    world = WorldModel(bus=bus)
    tools = ToolRegistry()
    reasoners: list[Reasoner] = [MathSpecialist()]
    RemoteRequestHandler(bus=bus, node_id=node_id, reasoners=reasoners,
                         world=world, tools=tools).wire()
    stop = asyncio.Event()
    ann = asyncio.create_task(
        Announcer(bus=bus, node_id=node_id, reasoners=reasoners,
                  specialization="math").run(stop)
    )
    print(f"[{node_id}] peer up on port {port}; Ctrl-C to stop.")
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop.set()
        ann.cancel()
        await bus.close()


# ---------------------------------------------------------------------------
# smoke tests
# ---------------------------------------------------------------------------

async def _smoke_tests() -> int:
    """Cheap end-to-end checks. Exits non-zero on failure."""
    from .bus import EventBus
    from .transport import TCPTransport
    from .types import Event

    print("== bus smoke test ==")
    port = 7999
    hub = EventBus(transport=TCPTransport(node_id="t.hub", port=port, role="hub"))
    spoke = EventBus(transport=TCPTransport(node_id="t.spoke", port=port, role="spoke"))
    await hub.start()
    await spoke.start()
    inbox_hub: list[Event] = []
    inbox_spoke: list[Event] = []
    async def on_hub(e: Event) -> None:
        inbox_hub.append(e)
    async def on_spoke(e: Event) -> None:
        inbox_spoke.append(e)
    hub.subscribe("t.*", on_hub)
    spoke.subscribe("t.*", on_spoke)
    await spoke.publish(Event(topic="t.ping", payload={}, origin="t.spoke"))
    await hub.publish(Event(topic="t.pong", payload={}, origin="t.hub"))
    await asyncio.sleep(0.1)
    assert any(e.origin == "t.spoke" for e in inbox_hub), "hub didn't receive"
    assert any(e.origin == "t.hub" for e in inbox_spoke), "spoke didn't receive"
    await spoke.close()
    await hub.close()
    print("  ✓ TCP bus round-trip")

    print("== synthesis smoke test ==")
    rt = Runtime(root=_data_root(".mente-test"))
    await rt.start()
    r = await rt.handle_intent(Intent(text="compute the 10th fibonacci number"))
    assert "55" in r.text, f"expected fib(10)=55, got: {r.text!r}"
    print(f"  ✓ synthesis: {r.text}")
    r = await rt.handle_intent(Intent(text="what is the factorial of 6"))
    assert "720" in r.text, f"expected factorial(6)=720, got: {r.text!r}"
    print(f"  ✓ synthesis: {r.text}")
    await rt.shutdown()

    print("== semantic memory smoke test ==")
    rt = Runtime(root=_data_root(".mente-test"))
    await rt.start()
    rt.semantic_mem.remember("redis uses AOF or RDB for persistence")
    rt.semantic_mem.remember("postgres uses write-ahead logging")
    hits = rt.semantic_mem.search("database durability", k=2)
    assert hits, "semantic search returned nothing"
    print(f"  ✓ semantic search top hit: {hits[0]['text']!r}")
    await rt.shutdown()

    shutil.rmtree(_data_root(".mente-test"), ignore_errors=True)
    print("\nall smoke tests passed.")
    return 0


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------


def _current_schema_version() -> int | None:
    """Return the current state/library schema version, or None if undefined.

    Unit 10 adds ``_SCHEMA_VERSION`` to ``mente.state`` / ``mente.synthesis``.
    We import defensively so this CLI still works if the attribute is absent
    — in that case every file is reported "already current".
    """
    from . import state, synthesis

    for mod in (state, synthesis):
        v = getattr(mod, "_SCHEMA_VERSION", None)
        if isinstance(v, int):
            return v
    return None


def _file_schema_version(payload: Any) -> int | None:
    """Best-effort extraction of a file's declared schema version.

    Matches the ``{"_schema": <int>, ...}`` envelope used by
    ``LatentState.checkpoint`` / ``LibraryStore.save``. A bare dict with no
    ``_schema`` key is pre-versioning (v0).
    """
    if isinstance(payload, dict):
        v = payload.get("_schema")
        if isinstance(v, int):
            return v
        return 0  # pre-versioning: bare {key: value} dict
    return None


def _looks_like_library(payload: dict[str, Any]) -> bool:
    """True for a LibraryStore file, versioned or pre-v1."""
    if "primitives" in payload:
        return True
    if "_schema" in payload or not payload:
        return False
    sample = next(iter(payload.values()), None)
    return isinstance(sample, dict) and "entrypoint" in sample and "source" in sample


def _looks_like_latent(payload: dict[str, Any]) -> bool:
    """True for a LatentState file, versioned or pre-v1 bare dict."""
    if "values" in payload and isinstance(payload.get("values"), dict):
        return True
    if "_schema" in payload:
        return False
    return not _looks_like_library(payload)


def _upgrade_payload(payload: Any, target: int) -> Any:
    """Drive the tolerant loaders for a known file, or stamp as a fallback.

    Dispatch by envelope shape: library files go through
    ``synthesis._migrate_library``, latent files through ``state._migrate``.
    Unknown shapes just get their ``_schema`` marker stamped — we don't
    invent migration logic.
    """
    from . import state, synthesis

    if not isinstance(payload, dict):
        return payload

    library_migrate = getattr(synthesis, "_migrate_library", None)
    latent_migrate = getattr(state, "_migrate", None)
    from_version = _file_schema_version(payload) or 0

    if _looks_like_library(payload) and callable(library_migrate):
        return library_migrate(payload, from_version=from_version)
    if _looks_like_latent(payload) and callable(latent_migrate):
        return latent_migrate(payload, from_version=from_version)

    stamped = dict(payload)
    stamped["_schema"] = target
    return stamped


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomic replacement matching state.py / synthesis.py conventions."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=str, indent=2))
    tmp.replace(path)


def _migrate(data_dir: str, *, dry_run: bool) -> int:
    """Walk ``data_dir`` and upgrade every ``.json`` to the current schema.

    Returns 0 on success, 1 if the data directory is missing.
    """
    log = get_logger("cli.migrate")
    root = _data_root(data_dir)
    if not root.exists():
        print(f"data dir {root} does not exist; nothing to migrate.")
        return 1

    target = _current_schema_version()
    inspected = upgraded = current = skipped = 0

    for path in sorted(root.rglob("*.json")):
        if not path.is_file():
            continue
        inspected += 1
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            skipped += 1
            log.warning("skipping %s: %s", path, e)
            print(f"  skip  {path} ({e})")
            continue

        from_version = _file_schema_version(payload)
        needs_upgrade = target is not None and from_version != target

        if not needs_upgrade:
            current += 1
            continue

        if dry_run:
            print(f"  would upgrade {path} from v{from_version} to v{target}")
            upgraded += 1
            continue

        try:
            new_payload = _upgrade_payload(payload, target)
            _atomic_write_json(path, new_payload)
        except OSError as e:
            skipped += 1
            log.warning("failed to write %s: %s", path, e)
            print(f"  skip  {path} (write failed: {e})")
            continue
        print(f"  upgrade {path}: v{from_version} -> v{target}")
        upgraded += 1

    print(
        f"\n{inspected} files inspected, {upgraded} upgraded, "
        f"{current} already current, {skipped} skipped (corrupt/unreadable)."
    )
    return 0


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def _reset() -> None:
    root = _root()
    removed = []
    for p in sorted(root.glob(".mente*")):
        shutil.rmtree(p, ignore_errors=True)
        removed.append(p.name)
    if removed:
        print(f"removed: {', '.join(removed)}")
    else:
        print("nothing to reset.")


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

async def _do_run(data_dir: str) -> None:
    rt = Runtime(root=_data_root(data_dir), config=MenteConfig.load())
    await rt.start()
    await _seed(rt)
    bg = rt.start_background()
    try:
        await _run_repl(rt)
    finally:
        rt.stop_background()
        await asyncio.gather(*bg, return_exceptions=True)
        await rt.shutdown()


async def _do_demo(data_dir: str) -> None:
    rt = Runtime(root=_data_root(data_dir), config=MenteConfig.load())
    await rt.start()
    await _seed(rt)
    bg = rt.start_background()
    try:
        await _run_demo(rt)
    finally:
        rt.stop_background()
        await asyncio.gather(*bg, return_exceptions=True)
        await rt.shutdown()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mente",
        description="Persistent, event-driven reasoning process. "
                    "Run with no args for an interactive REPL.",
    )
    sub = p.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="interactive REPL (default)")
    p_run.add_argument("--data", default=".mente", help="state directory")

    p_demo = sub.add_parser("demo", help="scripted walkthrough")
    p_demo.add_argument("--data", default=".mente", help="state directory")

    p_fed = sub.add_parser("federated", help="hub + peer in one process, real TCP bus")
    p_fed.add_argument("--port", type=int, default=7722)

    p_peer = sub.add_parser("peer", help="run only the math specialist peer (advanced)")
    p_peer.add_argument("--port", type=int, default=7722)
    p_peer.add_argument("--id", default="peer.math")

    sub.add_parser("test", help="smoke tests")
    sub.add_parser("reset", help="wipe all .mente* state directories")

    p_mig = sub.add_parser(
        "migrate",
        help="upgrade state JSON files under --data-dir to the current schema",
    )
    p_mig.add_argument("--data-dir", default=".mente", help="state directory")
    p_mig.add_argument(
        "--dry-run", action="store_true",
        help="report what would change without touching any files",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cmd = args.command or "run"

    def _install_sigint() -> None:
        # Let Ctrl-C cleanly exit asyncio loops on macOS.
        with contextlib.suppress(ValueError):
            signal.signal(signal.SIGINT, signal.default_int_handler)

    _install_sigint()

    if cmd == "run":
        asyncio.run(_do_run(args.data))
    elif cmd == "demo":
        asyncio.run(_do_demo(args.data))
    elif cmd == "federated":
        asyncio.run(_run_federated(args.port))
    elif cmd == "peer":
        asyncio.run(_run_peer_only(args.port, args.id))
    elif cmd == "test":
        return asyncio.run(_smoke_tests())
    elif cmd == "reset":
        _reset()
    elif cmd == "migrate":
        return _migrate(args.data_dir, dry_run=args.dry_run)
    else:
        build_parser().print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
