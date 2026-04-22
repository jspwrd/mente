"""Tests for mente.transport: InProcessTransport no-op + TCP hub/spoke flow."""
from __future__ import annotations

import asyncio
import contextlib

import pytest

from mente.transport import InProcessTransport, TCPTransport
from mente.types import Event
from tests.fixtures.core_events import EventCapture
from tests.fixtures.core_net import find_unused_port


async def _noop(_e: Event) -> None:
    return None


@pytest.mark.asyncio
async def test_inprocess_transport_start_and_close():
    t = InProcessTransport()
    cap = EventCapture()
    await t.start(cap.handler)
    await t.publish_remote(Event(topic="x", payload={}, origin="self"))
    await t.close()
    # In-process transport never delivers anything remotely.
    assert cap.events == []


@pytest.mark.asyncio
async def test_inprocess_publish_remote_is_noop():
    t = InProcessTransport()
    # No start, no handler. Must not raise.
    await t.publish_remote(Event(topic="x", payload={}, origin="self"))


@pytest.mark.asyncio
async def test_tcp_hub_and_spoke_roundtrip():
    port = find_unused_port()
    hub = TCPTransport(node_id="hub", port=port, role="hub")
    spoke = TCPTransport(node_id="spoke", port=port, role="spoke")

    hub_rx = EventCapture()
    spoke_rx = EventCapture()

    await hub.start(hub_rx.handler)
    try:
        await spoke.start(spoke_rx.handler)
        try:
            await spoke.publish_remote(Event(topic="s.1", payload={"v": 1}, origin="spoke"))
            await asyncio.sleep(0.1)
            assert "s.1" in hub_rx.topics()

            await hub.publish_remote(Event(topic="h.1", payload={"v": 2}, origin="hub"))
            await asyncio.sleep(0.1)
            assert "h.1" in spoke_rx.topics()
        finally:
            await spoke.close()
    finally:
        await hub.close()


@pytest.mark.asyncio
async def test_tcp_spoke_ignores_self_echo():
    """When the hub rebroadcasts a spoke's event, the spoke must drop it
    because origin matches its node_id."""
    port = find_unused_port()
    hub = TCPTransport(node_id="hub", port=port, role="hub")
    spoke_a = TCPTransport(node_id="a", port=port, role="spoke")
    spoke_b = TCPTransport(node_id="b", port=port, role="spoke")

    hub_rx = EventCapture()
    a_rx = EventCapture()
    b_rx = EventCapture()

    await hub.start(hub_rx.handler)
    try:
        await spoke_a.start(a_rx.handler)
        await spoke_b.start(b_rx.handler)
        await asyncio.sleep(0.05)  # let hub accept both spokes
        try:
            await spoke_a.publish_remote(Event(topic="fan.1", payload={}, origin="a"))
            await asyncio.sleep(0.15)

            assert "fan.1" in hub_rx.topics()
            assert "fan.1" in b_rx.topics()
            assert "fan.1" not in a_rx.topics()
        finally:
            await spoke_a.close()
            await spoke_b.close()
    finally:
        await hub.close()


@pytest.mark.asyncio
async def test_tcp_multiple_spokes_all_receive_hub_broadcast():
    port = find_unused_port()
    hub = TCPTransport(node_id="hub", port=port, role="hub")
    spoke_a = TCPTransport(node_id="a", port=port, role="spoke")
    spoke_b = TCPTransport(node_id="b", port=port, role="spoke")

    a_rx = EventCapture()
    b_rx = EventCapture()

    await hub.start(_noop)
    try:
        await spoke_a.start(a_rx.handler)
        await spoke_b.start(b_rx.handler)
        await asyncio.sleep(0.05)  # let hub register both writers
        try:
            await hub.publish_remote(Event(topic="bc.1", payload={}, origin="hub"))
            await asyncio.sleep(0.15)
            assert "bc.1" in a_rx.topics()
            assert "bc.1" in b_rx.topics()
        finally:
            await spoke_a.close()
            await spoke_b.close()
    finally:
        await hub.close()


@pytest.mark.asyncio
async def test_tcp_close_is_graceful_when_unstarted():
    t = TCPTransport(node_id="n", port=find_unused_port(), role="hub")
    await t.close()


@pytest.mark.asyncio
async def test_tcp_close_is_graceful_after_start():
    port = find_unused_port()
    hub = TCPTransport(node_id="hub", port=port, role="hub")
    spoke = TCPTransport(node_id="s", port=port, role="spoke")

    await hub.start(_noop)
    await spoke.start(_noop)
    await spoke.close()
    await hub.close()


@pytest.mark.asyncio
async def test_tcp_spoke_publish_without_connection_is_noop():
    """If a spoke is unstarted, publish_remote must silently drop."""
    t = TCPTransport(node_id="s", port=find_unused_port(), role="spoke")
    await t.publish_remote(Event(topic="x", payload={}, origin="s"))
    await t.close()


@pytest.mark.asyncio
async def test_tcp_event_fields_preserved_across_wire():
    port = find_unused_port()
    hub = TCPTransport(node_id="hub", port=port, role="hub")
    spoke = TCPTransport(node_id="spoke-xyz", port=port, role="spoke")

    hub_rx = EventCapture()
    await hub.start(hub_rx.handler)
    try:
        await spoke.start(_noop)
        try:
            await spoke.publish_remote(
                Event(
                    topic="chk",
                    payload={"k": "v", "n": 3},
                    origin="spoke-xyz",
                    confidence=0.42,
                )
            )
            await asyncio.sleep(0.1)
            got = next((e for e in hub_rx.events if e.topic == "chk"), None)
            assert got is not None, "hub did not receive event"
            assert got.origin == "spoke-xyz"
            assert got.payload == {"k": "v", "n": 3}
            assert got.confidence == 0.42
        finally:
            await spoke.close()
    finally:
        await hub.close()


@pytest.mark.asyncio
async def test_tcp_auth_handshake_round_trip_with_matching_secret():
    """Matching secret on hub + spoke: events flow end-to-end."""
    port = find_unused_port()
    hub = TCPTransport(node_id="hub", port=port, role="hub", auth_secret="s3cr3t")
    spoke = TCPTransport(node_id="spoke", port=port, role="spoke", auth_secret="s3cr3t")

    hub_rx = EventCapture()
    await hub.start(hub_rx.handler)
    try:
        await spoke.start(_noop)
        try:
            await spoke.publish_remote(Event(topic="auth.ok", payload={}, origin="spoke"))
            await asyncio.sleep(0.1)
            assert "auth.ok" in hub_rx.topics()
        finally:
            await spoke.close()
    finally:
        await hub.close()


@pytest.mark.asyncio
async def test_tcp_auth_handshake_rejects_wrong_secret():
    """Spoke with the wrong secret must not be able to deliver events."""
    port = find_unused_port()
    hub = TCPTransport(node_id="hub", port=port, role="hub", auth_secret="correct")
    spoke = TCPTransport(node_id="spoke", port=port, role="spoke", auth_secret="WRONG")

    hub_rx = EventCapture()
    await hub.start(hub_rx.handler)
    try:
        # Connecting succeeds at the TCP layer; the hub drops the stream after
        # the handshake fails, so the subsequent publish never lands.
        await spoke.start(_noop)
        try:
            await spoke.publish_remote(Event(topic="auth.bad", payload={}, origin="spoke"))
            await asyncio.sleep(0.15)
            assert "auth.bad" not in hub_rx.topics()
        finally:
            await spoke.close()
    finally:
        await hub.close()


@pytest.mark.asyncio
async def test_tcp_auth_handshake_rejects_missing_secret():
    """Hub requires a secret; spoke with no secret is rejected."""
    port = find_unused_port()
    hub = TCPTransport(node_id="hub", port=port, role="hub", auth_secret="needed")
    spoke = TCPTransport(node_id="spoke", port=port, role="spoke")  # no secret

    hub_rx = EventCapture()
    await hub.start(hub_rx.handler)
    try:
        await spoke.start(_noop)
        try:
            await spoke.publish_remote(Event(topic="auth.none", payload={}, origin="spoke"))
            await asyncio.sleep(0.15)
            assert "auth.none" not in hub_rx.topics()
        finally:
            await spoke.close()
    finally:
        await hub.close()


@pytest.mark.asyncio
async def test_tcp_hub_rebroadcast_excludes_sender():
    """Two spokes; one sends; only the other should receive (sender is excluded)."""
    port = find_unused_port()
    hub = TCPTransport(node_id="hub", port=port, role="hub")
    sender = TCPTransport(node_id="sender", port=port, role="spoke")
    receiver = TCPTransport(node_id="receiver", port=port, role="spoke")

    sender_rx = EventCapture()
    receiver_rx = EventCapture()

    await hub.start(_noop)
    try:
        await sender.start(sender_rx.handler)
        await receiver.start(receiver_rx.handler)
        await asyncio.sleep(0.05)  # let hub register both writers
        try:
            await sender.publish_remote(Event(topic="one", payload={}, origin="sender"))
            await asyncio.sleep(0.15)
            assert "one" in receiver_rx.topics()
            assert "one" not in sender_rx.topics()
        finally:
            await sender.close()
            await receiver.close()
    finally:
        await hub.close()


# ---------------------------------------------------------------------------
# Auto-reconnect tests
#
# These tests simulate an abrupt hub death by spinning up a bare asyncio
# server in the test (rather than a full TCPTransport hub) so we can tear
# it down and restart it on the same port without fighting the hub's own
# shutdown semantics. The wire protocol is the same JSON-lines format, so
# the spoke sees identical bytes either way.
# ---------------------------------------------------------------------------


class _BareHub:
    """Minimal hub stand-in: reads AUTH (if secret set), then echoes JSON
    lines to every connected spoke. Kills all connections on stop()."""

    def __init__(self, port: int, auth_secret: str | None = None) -> None:
        self.port = port
        self.auth_secret = auth_secret
        self.server: asyncio.base_events.Server | None = None
        self.writers: list[asyncio.StreamWriter] = []
        self.auths_seen: list[str] = []
        self.lines_rx: list[bytes] = []

    async def start(self) -> None:
        async def _handle(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            if self.auth_secret is not None:
                first = await reader.readline()
                self.auths_seen.append(first.decode().strip())
                # Accept any non-empty line; real hub verifies HMAC, but for
                # this test the presence of the AUTH frame is what matters.
                if not first.startswith(b"AUTH "):
                    writer.close()
                    return
            self.writers.append(writer)
            try:
                while line := await reader.readline():
                    self.lines_rx.append(line)
            except (ConnectionError, BrokenPipeError, OSError):
                pass
            finally:
                if writer in self.writers:
                    self.writers.remove(writer)

        self.server = await asyncio.start_server(_handle, "127.0.0.1", self.port)

    async def broadcast(self, line: bytes) -> None:
        for w in list(self.writers):
            try:
                w.write(line)
                await w.drain()
            except (ConnectionError, BrokenPipeError):
                pass

    async def stop(self) -> None:
        # Abort live clients first so close()/wait_closed don't hang on them.
        for w in list(self.writers):
            with contextlib.suppress(Exception):
                w.close()
        self.writers.clear()
        if self.server is not None:
            self.server.close()
            if hasattr(self.server, "abort_clients"):
                self.server.abort_clients()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.server.wait_closed(), timeout=1.0)
            self.server = None


@pytest.mark.asyncio
async def test_tcp_spoke_reconnects_after_hub_restart(monkeypatch):
    """Hub goes down mid-conversation; spoke must reconnect to a restarted hub
    on the same port and resume receiving events."""
    from mente import transport as transport_mod

    monkeypatch.setattr(transport_mod, "_RECONNECT_BACKOFF_S", (0.05,))
    monkeypatch.setattr(transport_mod, "_RECONNECT_MAX_ATTEMPTS", 30)

    port = find_unused_port()
    hub1 = _BareHub(port)
    await hub1.start()

    spoke_rx = EventCapture()
    spoke = TCPTransport(node_id="spoke", port=port, role="spoke")
    await spoke.start(spoke_rx.handler)
    await asyncio.sleep(0.05)

    # Sanity: hub-to-spoke delivery works.
    await hub1.broadcast(b'{"topic": "pre", "payload": {}, "origin": "hub"}\n')
    await asyncio.sleep(0.1)
    assert "pre" in spoke_rx.topics()

    # Kill the hub.
    await hub1.stop()
    await asyncio.sleep(0.1)
    assert spoke._hub_writer is None

    # New hub on the same port.
    hub2 = _BareHub(port)
    await hub2.start()

    for _ in range(40):
        await asyncio.sleep(0.05)
        if spoke._hub_writer is not None:
            break
    assert spoke._hub_writer is not None, "spoke did not reconnect within budget"
    assert spoke.reconnect_attempts == 0  # reset on success

    await hub2.broadcast(b'{"topic": "post", "payload": {}, "origin": "hub"}\n')
    await asyncio.sleep(0.15)
    assert "post" in spoke_rx.topics()

    await spoke.close()
    await hub2.stop()


@pytest.mark.asyncio
async def test_tcp_spoke_gives_up_cleanly_when_hub_stays_down(monkeypatch):
    """If the hub never comes back, the spoke gives up after the retry limit
    and close() returns promptly without hanging."""
    from mente import transport as transport_mod

    monkeypatch.setattr(transport_mod, "_RECONNECT_BACKOFF_S", (0.02,))
    monkeypatch.setattr(transport_mod, "_RECONNECT_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(transport_mod, "_RECONNECT_PROBE_TIMEOUT_S", 0.1)

    port = find_unused_port()
    hub = _BareHub(port)
    await hub.start()

    spoke = TCPTransport(node_id="spoke", port=port, role="spoke")
    await spoke.start(_noop)
    await asyncio.sleep(0.05)

    # Kill the hub permanently.
    await hub.stop()

    # Reconnect loop exhausts in ~3 * (0.02 + 0.1) ≈ 0.4s.
    await asyncio.sleep(0.8)

    # close() must not hang.
    await asyncio.wait_for(spoke.close(), timeout=2.0)


@pytest.mark.asyncio
async def test_tcp_spoke_close_interrupts_reconnect_backoff(monkeypatch):
    """Calling close() while the spoke is mid-backoff must return promptly."""
    from mente import transport as transport_mod

    monkeypatch.setattr(transport_mod, "_RECONNECT_BACKOFF_S", (5.0,))
    monkeypatch.setattr(transport_mod, "_RECONNECT_MAX_ATTEMPTS", 10)

    port = find_unused_port()
    hub = _BareHub(port)
    await hub.start()

    spoke = TCPTransport(node_id="spoke", port=port, role="spoke")
    await spoke.start(_noop)
    await asyncio.sleep(0.05)

    await hub.stop()
    # Let the spoke enter backoff.
    await asyncio.sleep(0.1)

    loop = asyncio.get_event_loop()
    t0 = loop.time()
    await asyncio.wait_for(spoke.close(), timeout=2.0)
    elapsed = loop.time() - t0
    assert elapsed < 2.0, f"close took {elapsed:.2f}s, backoff was 5s"


@pytest.mark.asyncio
async def test_tcp_spoke_resends_auth_on_reconnect(monkeypatch):
    """After reconnect, the spoke must re-send the auth handshake frame."""
    from mente import transport as transport_mod

    monkeypatch.setattr(transport_mod, "_RECONNECT_BACKOFF_S", (0.05,))
    monkeypatch.setattr(transport_mod, "_RECONNECT_MAX_ATTEMPTS", 30)

    port = find_unused_port()
    hub1 = _BareHub(port, auth_secret="shared")
    await hub1.start()

    spoke = TCPTransport(
        node_id="spoke", port=port, role="spoke", auth_secret="shared"
    )
    await spoke.start(_noop)
    await asyncio.sleep(0.1)
    assert len(hub1.auths_seen) == 1
    assert hub1.auths_seen[0].startswith("AUTH spoke")

    await hub1.stop()
    await asyncio.sleep(0.05)

    # Fresh hub. If the spoke doesn't re-send AUTH, hub2.auths_seen stays empty.
    hub2 = _BareHub(port, auth_secret="shared")
    await hub2.start()

    for _ in range(60):
        await asyncio.sleep(0.05)
        if spoke._hub_writer is not None and hub2.auths_seen:
            break

    assert spoke._hub_writer is not None, "spoke did not reconnect"
    assert len(hub2.auths_seen) == 1, "spoke did not re-send AUTH on reconnect"
    assert hub2.auths_seen[0].startswith("AUTH spoke")

    await spoke.close()
    await hub2.stop()


@pytest.mark.asyncio
async def test_tcp_spoke_publish_during_reconnect_is_noop(monkeypatch):
    """During the reconnect gap, publish_remote must silently drop."""
    from mente import transport as transport_mod

    monkeypatch.setattr(transport_mod, "_RECONNECT_BACKOFF_S", (0.5,))
    monkeypatch.setattr(transport_mod, "_RECONNECT_MAX_ATTEMPTS", 5)

    port = find_unused_port()
    hub = _BareHub(port)
    await hub.start()

    spoke = TCPTransport(node_id="spoke", port=port, role="spoke")
    await spoke.start(_noop)
    await asyncio.sleep(0.05)

    await hub.stop()
    await asyncio.sleep(0.1)

    await spoke.publish_remote(Event(topic="dropped", payload={}, origin="spoke"))
    await spoke.close()


@pytest.mark.asyncio
async def test_tcp_spoke_reconnect_attempts_counter_increments(monkeypatch):
    """The read-only `reconnect_attempts` counter reflects attempts while the
    hub is down, then resets to 0 on a successful reconnect."""
    from mente import transport as transport_mod

    monkeypatch.setattr(transport_mod, "_RECONNECT_BACKOFF_S", (0.02,))
    monkeypatch.setattr(transport_mod, "_RECONNECT_MAX_ATTEMPTS", 20)
    monkeypatch.setattr(transport_mod, "_RECONNECT_PROBE_TIMEOUT_S", 0.05)

    port = find_unused_port()
    hub = _BareHub(port)
    await hub.start()

    spoke = TCPTransport(node_id="spoke", port=port, role="spoke")
    await spoke.start(_noop)
    await asyncio.sleep(0.05)
    assert spoke.reconnect_attempts == 0

    await hub.stop()
    # Let at least one reconnect attempt fire while the hub is down.
    await asyncio.sleep(0.2)
    assert spoke.reconnect_attempts >= 1

    await spoke.close()
