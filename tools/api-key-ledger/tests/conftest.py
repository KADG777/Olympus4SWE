from pathlib import Path

import pytest

from key_ledger.config import Config, Credential, resolve_credentials

ENV = {
    "LEDGER_TEST_A": "test-credential-env-alpha",
    "LEDGER_TEST_B": "test-credential-env-bravo",
    "LEDGER_TEST_C": "test-credential-env-charlie",
    "LEDGER_TEST_D": "test-credential-env-delta",
}
PRICE = {
    "currency": "USD",
    "input_per_1m": "2",
    "output_per_1m": "4",
    "cached_input_per_1m": "0.5",
    "cache_write_per_1m": "3",
}


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config.model_validate(
        {
            "database": tmp_path / "ledger.sqlite3",
            "pools": {
                "chat": {
                    "dialect": "openai",
                    "base_url": "https://chat.example.test/gateway/v1/",
                    "keys": [
                        {"key_alias": "a", "secret": "env:LEDGER_TEST_A"},
                        {"key_alias": "b", "secret": "env:LEDGER_TEST_B"},
                    ],
                    "pricing": {"mock-model": PRICE},
                },
                "messages": {
                    "dialect": "anthropic",
                    "base_url": "https://messages.example.test",
                    "keys": [
                        {"key_alias": "c", "secret": "env:LEDGER_TEST_C"},
                        {"key_alias": "d", "secret": "env:LEDGER_TEST_D"},
                    ],
                    "pricing": {"mock-model": PRICE},
                },
            },
        }
    )


@pytest.fixture
def credentials(config: Config) -> dict[tuple[str, str], Credential]:
    return resolve_credentials(config, ENV)
