"""Global test safeguards."""

import socket
from collections.abc import Iterator

import pytest
from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: MonkeyPatch) -> Iterator[None]:
    """Fail every real socket connection; providers must be mocked or fake in tests."""

    def blocked_connect(*_: object, **__: object) -> None:
        raise AssertionError("external network access is forbidden during pytest")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    yield
