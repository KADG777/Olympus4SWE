"""Standalone command line interface."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import uvicorn

from .config import DEFAULT_CONFIG, ConfigError, load_config, resolve_credentials
from .ledger import GROUP_COLUMNS, Ledger
from .proxy import create_app, parse_since

SUMMARY_FIELDS = (
    "group",
    "currency",
    "attempts",
    "successes",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "cost",
    "unpriced",
    "unknown_usage",
)
KEY_FIELDS = (
    "pool_id",
    "key_alias",
    "status",
    "fingerprint",
    "selections",
    "cooldown_until",
    "currency",
    "daily_spend",
    "total_spend",
    "budget",
    "pool_daily_spend",
    "pool_total_spend",
    "pool_budget",
)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)


def _write(rows: list[dict[str, Any]], fields: tuple[str, ...], fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    values = [[_cell(row.get(name)) for name in fields] for row in rows]
    if fmt == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(fields)
        for row in values:
            writer.writerow(
                [
                    "'" + value
                    if value.lstrip().startswith(("=", "+", "-", "@"))
                    or value.startswith(("\t", "\r", "\n"))
                    else value
                    for value in row
                ]
            )
        return
    widths = [max(len(cell) for cell in column) for column in zip(fields, *values, strict=True)]
    for entry in [fields, *values]:
        print(
            "  ".join(cell.ljust(width) for cell, width in zip(entry, widths, strict=True)).rstrip()
        )


def _port(raw: str) -> int:
    value = int(raw)
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="key-ledger", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="start the proxy and read-only dashboard")
    serve.add_argument("--host", choices=("127.0.0.1", "::1"), default="127.0.0.1")
    serve.add_argument("--port", type=_port, default=8787)

    keys = commands.add_parser("keys", help="inspect or recover configured key slots")
    key_commands = keys.add_subparsers(dest="key_command", required=True)
    listing = key_commands.add_parser("list")
    listing.add_argument("--format", choices=("table", "json", "csv"), default="table")
    for name in ("enable", "disable", "reset"):
        command = key_commands.add_parser(name)
        command.add_argument("pool")
        command.add_argument("key_alias")

    summary = commands.add_parser("summary", help="aggregate attempts and recorded spend")
    summary.add_argument("--by", choices=tuple(GROUP_COLUMNS), default="model")
    summary.add_argument("--since", help="ISO date or timestamp (UTC by default)")
    summary.add_argument("--format", choices=("table", "json", "csv"), default="table")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "serve":
            credentials = resolve_credentials(config)
            uvicorn.run(
                create_app(config, credentials),
                host=args.host,
                port=args.port,
                access_log=False,
                proxy_headers=False,
                server_header=False,
                timeout_graceful_shutdown=15,
            )
            return 0

        ledger = Ledger(config)
        if args.command == "summary":
            try:
                since = parse_since(args.since) if args.since else 0
            except (ValueError, OverflowError, OSError):
                parser.error("--since must be an ISO date or timestamp")
            _write(ledger.summary(args.by, since), SUMMARY_FIELDS, args.format)
        elif args.key_command == "list":
            _write(ledger.key_statuses(), KEY_FIELDS, args.format)
        else:
            ledger.change_key(args.pool, args.key_alias, args.key_command)
        return 0
    except ConfigError as exc:
        print(f"key-ledger: {exc}", file=sys.stderr)
    except (sqlite3.Error, OSError):
        print("key-ledger: could not open local ledger storage", file=sys.stderr)
    except ValueError:
        print("key-ledger: invalid pool or key alias", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
