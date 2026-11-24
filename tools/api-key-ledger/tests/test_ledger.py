import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from key_ledger.cli import main
from key_ledger.config import (
    Budget,
    Config,
    ConfigError,
    Credential,
    load_config,
    resolve_credentials,
)
from key_ledger.ledger import RESERVATION_TTL, Attempt, Ledger
from key_ledger.usage import Usage


def reserve(ledger: Ledger, sticky: str | None = None) -> Attempt:
    attempt, _ = ledger.reserve(
        "chat", model="mock-model", sticky_key=sticky, tag="pytest", tried=set()
    )
    assert attempt is not None
    return attempt


def finish(ledger: Ledger, attempt: Attempt, *, count: int = 10, outcome: str = "success") -> None:
    price = ledger.config.pools["chat"].pricing["mock-model"]
    ledger.finish(
        attempt,
        usage=Usage("mock-model", uncached=count, output=0, cache_read=0, cache_write=0),
        price=price,
        outcome=outcome,
        http_status=200,
    )


@pytest.mark.parametrize("rotation", ["round_robin", "least_used", "least_cost"])
def test_rotation_sticky_and_restart(
    config: Config,
    credentials: dict[tuple[str, str], Credential],
    rotation: str,
) -> None:
    config.pools["chat"].rotation = rotation
    ledger = Ledger(config, credentials)
    first = reserve(ledger, "scope-1")
    assert first.key_alias == "a"
    finish(ledger, first, count=1_000)
    second = reserve(ledger, "scope-1")
    assert second.key_alias == "a"
    finish(ledger, second, count=1_000)

    restarted = Ledger(config, credentials)
    third = reserve(restarted, "scope-2")
    assert third.key_alias == "b"
    finish(restarted, third, count=1)
    fourth = reserve(restarted)
    assert fourth.key_alias == ("a" if rotation == "round_robin" else "b")
    finish(restarted, fourth, count=1)
    assert len(restarted.events()) == 4


def test_atomic_capped_reservation_and_immutable_history(
    config: Config,
    credentials: dict[tuple[str, str], Credential],
) -> None:
    config.pools["chat"].budget = Budget(daily="0.00002")
    ledgers = [Ledger(config, credentials), Ledger(config, credentials)]

    def try_select(ledger: Ledger) -> Attempt | None:
        return ledger.reserve("chat", model="mock-model", sticky_key=None, tag=None, tried=set())[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        attempts = list(pool.map(try_select, ledgers))
    (winning,) = [attempt for attempt in attempts if attempt is not None]
    finish(ledgers[0], winning)
    assert try_select(ledgers[1]) is None
    assert {row["status"] for row in ledgers[0].key_statuses() if row["pool_id"] == "chat"} == {
        "budget_exceeded"
    }
    (event,) = ledgers[0].events()
    config.pools["chat"].pricing["mock-model"].input_per_1m *= 10
    assert Ledger(config, credentials).events() == [event]
    assert event["cost"] == "0.00002"
    assert event["price_snapshot"]["input_per_1m"] == "2"
    ledgers[0].change_key("chat", winning.key_alias, "reset")
    assert try_select(ledgers[0]) is None


def test_stale_attempt_recovery_and_cli(
    config: Config,
    credentials: dict[tuple[str, str], Credential],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger = Ledger(config, credentials)
    attempt = reserve(ledger)
    with sqlite3.connect(config.database) as db:
        db.execute("UPDATE reservations SET started_at=started_at-?", (RESERVATION_TTL + 1,))
    restarted = Ledger(config, credentials)
    (old,) = restarted.events()
    assert old["outcome"] == "interrupted"
    assert old["cost"] is None
    finish(ledger, attempt)
    assert restarted.events() == [old]

    winning = reserve(restarted)
    finish(restarted, winning)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    command = ["--config", str(config_file)]
    for by in ("key", "model", "provider", "day", "tag"):
        for fmt in ("json", "csv", "table"):
            assert main([*command, "summary", "--by", by, "--format", fmt]) == 0
            output = capsys.readouterr().out
            if fmt == "json":
                rows = json.loads(output)
                assert sum(row["attempts"] for row in rows) == 2
                assert {row["cost"] for row in rows} & {"0.00002"}
            else:
                assert "attempts" in output and "0.00002" in output
    assert main([*command, "summary", "--since", "2100-01-01", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out) == []

    for action in ("disable", "reset", "enable"):
        assert main([*command, "keys", action, "chat", winning.key_alias]) == 0
        assert main([*command, "keys", "list", "--format", "json"]) == 0
        (row,) = [
            row
            for row in json.loads(capsys.readouterr().out)
            if row["key_alias"] == winning.key_alias
        ]
        assert row["status"] == ("active" if action == "enable" else "disabled")


def test_cooldown_and_daily_limit_reopen(
    config: Config,
    credentials: dict[tuple[str, str], Credential],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config.pools["chat"].keys[0].budget = Budget(daily="0.00002")
    ledger = Ledger(config, credentials)
    epoch = 1_800_000_000
    monkeypatch.setattr("key_ledger.ledger.time.time", lambda: epoch)
    first = reserve(ledger)
    finish(ledger, first)
    second = reserve(ledger)
    ledger.finish(
        second, usage=Usage("mock-model"), price=None, outcome="rate_limited", cooldown=60
    )
    assert ledger.reserve("chat", model=None, sticky_key=None, tag=None, tried=set())[0] is None
    epoch += 61
    assert reserve(ledger).key_alias == "b"
    epoch += 86400
    assert {row["status"] for row in ledger.key_statuses()} == {"active"}


def test_config_errors_mask_raw_input(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    secret = "secret-should-never-appear"
    config_file.write_text(f"pools: {{bad: {{secret: {secret}}}}}", encoding="utf-8")
    with pytest.raises(ConfigError) as failure:
        load_config(config_file)
    assert secret not in str(failure.value)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission enforcement")
def test_secrets_file_permissions(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.secrets_file = tmp_path / "credentials.json"
    config.secrets_file.write_text('{"KEY":"file-only-secret"}', encoding="utf-8")
    config.pools["chat"].keys[0].secret = "file:KEY"
    env = {
        "LEDGER_TEST_B": "test-key-2",
        "LEDGER_TEST_C": "test-key-3",
        "LEDGER_TEST_D": "test-key-4",
    }
    config.secrets_file.chmod(0o644)
    with pytest.raises(ConfigError, match="0600"):
        resolve_credentials(config, env)
    config.secrets_file.chmod(0o600)
    resolved = resolve_credentials(config, env)
    assert resolved["chat", "a"].value == "file-only-secret"
    assert "file-only-secret" not in repr(resolved)
    link = tmp_path / "credentials-link.json"
    link.symlink_to(config.secrets_file)
    monkeypatch.setattr(config, "secrets_file", link)
    with pytest.raises(ConfigError, match="cannot read"):
        resolve_credentials(config, env)


@pytest.mark.skipif(os.name == "posix", reason="non-POSIX fallback")
def test_non_posix_secrets_fail_closed(config: Config) -> None:
    config.pools["chat"].keys[0].secret = "file:KEY"
    with pytest.raises(ConfigError, match="use env references"):
        resolve_credentials(config, {})
