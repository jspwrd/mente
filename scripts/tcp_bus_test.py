"""Minimal two-peer TCP bus smoke test.

Spawns a hub and a spoke on the same loop, has each publish an event, and
verifies the other receives it. If this works, multi-process peers across
the LAN work too.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aria.bus import EventBus
from aria.transport import TCPTransport
from aria.types import Event


async def main() -> int:
    port = 7790

    hub_bus = EventBus(transport=TCPTransport(node_id="hub", host="127.0.0.1", port=port, role="hub"))
    await hub_bus.start()

    spoke_bus = EventBus(transport=TCPTransport(node_id="spoke", host="127.0.0.1", port=port, role="spoke"))
    await spoke_bus.start()

    hub_inbox: list[Event] = []
    spoke_inbox: list[Event] = []

    async def hub_recv(e: Event) -> None:
        hub_inbox.append(e)
    async def spoke_recv(e: Event) -> None:
        spoke_inbox.append(e)

    hub_bus.subscribe("peer.*", hub_recv, name="hub.recv")
    spoke_bus.subscribe("peer.*", spoke_recv, name="spoke.recv")

    # Spoke publishes → should reach hub.
    await spoke_bus.publish(Event(topic="peer.ping", payload={"from": "spoke"}, origin="spoke"))
    # Hub publishes → should reach spoke.
    await hub_bus.publish(Event(topic="peer.pong", payload={"from": "hub"}, origin="hub"))

    await asyncio.sleep(0.1)  # let messages settle

    assert any(e.origin == "spoke" for e in hub_inbox), f"hub missed spoke msg: {hub_inbox}"
    assert any(e.origin == "hub" for e in spoke_inbox), f"spoke missed hub msg: {spoke_inbox}"

    print(f"hub received {len(hub_inbox)} events; spoke received {len(spoke_inbox)} events")
    print("TCP bus OK.")

    await spoke_bus.close()
    await hub_bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
