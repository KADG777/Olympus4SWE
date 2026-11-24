# API Key Ledger

A standalone loopback gateway for development and testing. It rotates upstream API keys, records
each attempt in SQLite WAL, and serves a read-only dashboard. It accepts OpenAI Chat Completions and
Anthropic Messages in JSON and SSE modes. The tool has its own Python package and dependencies.

## Start the gateway

From this directory, copy `config.example.yaml` to `config.yaml`. Set each pool's upstream
`base_url`, dialect, non-secret key aliases, and secret references. Replace the example model IDs
and rates with your provider's prices. The example rates are for illustration.

Set the environment variables referenced by your pools in the same terminal. For example, in
PowerShell 7, for a chat-only configuration with two keys:

```powershell
$env:LEDGER_CHAT_A = Read-Host 'Chat key A' -MaskInput
$env:LEDGER_CHAT_B = Read-Host 'Chat key B' -MaskInput
uv run --locked key-ledger --config config.yaml serve
```

The sample also defines a Messages pool. Set its two environment variables as well, or remove that
pool and its `default_pools` entry from your local config. The default config path is
`~/.config/olympus-key-ledger/config.yaml`. Relative paths in a config resolve from its directory.

Use `env:VARIABLE` for Windows or POSIX. On POSIX, `file:NAME` resolves an entry in a JSON file
that maps secret names to values. The default is `~/.config/olympus-key-ledger/secrets.json`.
Create a regular file owned by the launch user with mode `0600` before writing values. On Windows,
use environment references. Store credentials outside git.

The proxy listens on `127.0.0.1:8787` by default. `serve --port 8790` selects a port,
and `serve --host ::1` selects IPv6 loopback. Point your SDK or CLI at the appropriate route.

| Caller | Default pool | Named pool |
| --- | --- | --- |
| `OPENAI_BASE_URL` | `http://127.0.0.1:8787/v1` | `http://127.0.0.1:8787/p/chat/v1` |
| `ANTHROPIC_BASE_URL` | `http://127.0.0.1:8787` | `http://127.0.0.1:8787/p/messages` |

Set a local placeholder such as `ledger-local` if the caller SDK requires a key. The proxy
constructs upstream authentication with the selected slot. With multiple pools for a dialect,
set its ID in `default_pools` or use the named route. A single matching pool becomes the default
automatically. For OpenAI-compatible pools, a host-only upstream URL defaults to `/v1`.
A configured path such as `/api/v3` or `/v1beta/openai` is used as the SDK base path.

Optional `X-Ledger-Sticky-Key` pins a scope to a healthy slot. Use a short non-secret identifier
such as a test case ID. SQLite stores its SHA-256 digest. Optional `X-Ledger-Tag` adds
an attribution label such as `pytest` or `eval-v1`. Tags support letters, numbers, and
`_.:/@+-`, must begin with a letter or number, and have a 160-character limit.

## Rotation and limits

- `round_robin` follows config order and persists its cursor. Sticky hits preserve the cursor.
  `least_used` uses recorded selections. `least_cost` uses lifetime known spend, with selection
  count as the tiebreaker. An unavailable pin moves to another healthy slot.
- A pre-stream `429` honors `Retry-After` or uses the pool's cooldown. A pre-stream
  `401`, `402`, or `403` quarantines the slot. The gateway tries another eligible slot once per
  request and records each attempt. Exhaustion returns structured `503`, including slot counts
  by availability. Other upstream HTTP errors preserve their status and a credential-redacted
  body of up to 64 KiB. Oversized or unreadable bodies get a generic error message. A stream
  that has started remains on its original slot and records partial usage on error or disconnect.
- Slot and pool budgets can define `daily` and/or `total` caps in the pool currency. The day
  rolls over at midnight UTC. Each attempt checks recorded spend at admission. Multiple
  attempts may run concurrently while recorded spend is below the cap, and their final charges
  may together cross it. Once recorded spend reaches the cap, subsequent requests skip the
  affected slots. Unpriced usage and missing upstream usage have unknown cost and are surfaced
  separately.
- Price snapshots are immutable. Tokens include total input, output, cached input reads, and
  cache writes. Cached reads and writes are subsets of the total input. Unknown components
  remain null in event records. The gateway uses the returned model ID when available.
  OpenAI-compatible pools can set `inject_stream_usage: false` if their gateway rejects
  `stream_options`. The gateway records a known partial cost when a stream drops early.
- Use trusted upstream endpoints. Upstream replies are streamed to the requesting client.
  The local tool accepts loopback hosts, rejects cross-origin requests, and keeps the
  dashboard read-only. Local processes that can reach the loopback port can use the proxy.
  The gateway relays the two documented endpoints and accepts up to 8 MiB of JSON per request.

## Inspect usage

Open [the dashboard](http://127.0.0.1:8787/dashboard) for charts, key state and recent attempts.
Its read-only API includes `GET /health`, `GET /api/keys`, `GET /api/events?limit=50`, and
`GET /api/stats?by=model&since=2026-01-01`.

```text
uv run --locked key-ledger --config config.yaml keys list
uv run --locked key-ledger --config config.yaml keys disable chat chat-a
uv run --locked key-ledger --config config.yaml keys enable chat chat-a
uv run --locked key-ledger --config config.yaml keys reset chat chat-a
uv run --locked key-ledger --config config.yaml summary --by tag --since 2026-01-01 --format csv
```

`enable` clears manual disable, quarantine, and cooldown. `reset` clears quarantine and cooldown
and respects manual disable. Both preserve spend and history. Summary groups are `key`, `model`,
`provider` (pool ID), `day`, and `tag`. Use `table`, `json`, or `csv`. Currency totals stay
separate. Rates use up to six decimal places per million tokens, and ledger costs use fixed-point
arithmetic with up to 12 decimal places in the pool currency.

## Verify

```text
uv sync --locked
uv run --locked --offline pytest -q
uv run --locked --offline ruff check .
uv run --locked --offline mypy src
```

The tests use simulated upstreams and temporary SQLite databases.
