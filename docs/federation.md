# Federation

MENTE's bus can stretch across machines. This page is a practical guide to a
two-host deployment: one laptop running the coordinator, one GPU box running a
heavy reasoner, both on the same LAN (or a VPN like Tailscale / ZeroTier).

## What federation is good for

- Keep a deep, slow reasoner pinned to the box with the GPU while the laptop
  hosts the coordinator, fast tiers, memory, and REPL.
- Share one specialist across several front-ends — a single math or code
  specialist can serve every peer that discovers it.
- Colocate reasoners with the data they need (private docs, local LLM, etc.)
  without shipping secrets off-box.

Federation is not a replacement for a job queue. It is a pub/sub seam: every
peer sees the announcement topic, and direct requests are routed by
`target_node`.

## Minimal setup

You need:

- Both hosts reachable over TCP on one port (default `7722`).
- `MENTE_BUS_SECRET` set to the *same* string on both sides. Any non-empty
  value turns on the HMAC handshake; an empty/unset value disables auth
  entirely — fine for localhost, not fine for a VPN.
- One host designated `hub` (binds the port). Every other host is a `spoke`
  and connects to the hub.

The env vars are read by `tcp_from_env` in `mente.transport`: `MENTE_BUS_ROLE`,
`MENTE_BUS_HOST`, `MENTE_BUS_PORT`, `MENTE_BUS_SECRET`. Nothing else is new.

## Concrete walkthrough (Tailscale)

Install Tailscale on both hosts and bring it up. On the hub:

```bash
tailscale ip -4        # e.g. 100.88.12.34
```

Start the hub:

```bash
export MENTE_BUS_SECRET=$(head -c32 /dev/urandom | base64)
MENTE_BUS_ROLE=hub MENTE_BUS_PORT=7722 mente run
```

Copy the secret to the spoke's shell (paste into `MENTE_BUS_SECRET`) and
start the spoke:

```bash
MENTE_BUS_ROLE=spoke \
MENTE_BUS_HOST=100.88.12.34 \
MENTE_BUS_PORT=7722 \
MENTE_BUS_SECRET=<same-secret> \
mente peer
```

Within one announcement interval (2s) each side's `Directory` contains the
other's reasoners, and the coordinator's router will dispatch matching intents
over the bus.

## Troubleshooting

- **Firewall.** On Ubuntu: `sudo ufw allow 7722/tcp`. Without this the spoke
  hangs on `open_connection`.
- **Wrong secret.** The hub logs `transport.auth.reject reason=bad-hmac` and
  drops the spoke. Check both shells — secrets with embedded `$` in a script
  are the usual culprit.
- **Clock skew.** The handshake rejects timestamps more than `±60s` out of
  date (`_AUTH_MAX_SKEW_S`). Run `chronyc tracking` or equivalent if the two
  hosts disagree.
- **Stale peer.** A spoke that crashed won't vanish from the hub's directory
  immediately; the periodic announcement is the liveness signal. Spokes that
  never re-announce are evicted by the background sweep.

## What breaks

- One peer disappearing — the remaining nodes keep serving their own
  reasoners; in-flight remote requests to the dead peer time out and return a
  fallback response.
- Wrong secret on either side — connection refused at handshake time, no
  events cross.
- Clock drift past the skew window — same as wrong secret.
- VPN MTU too small — JSON lines longer than the path MTU can wedge. If
  `publish_remote` stalls, drop Tailscale MTU to 1280 on both ends.
