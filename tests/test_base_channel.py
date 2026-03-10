# Tests for BaseChannel.is_allowed: exact-match semantics to prevent
# injection attacks via pipe-separated sender IDs.

from types import SimpleNamespace

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel


# Minimal channel implementation for testing base class behavior
class _DummyChannel(BaseChannel):
    name = "dummy"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(self, msg: OutboundMessage) -> None:
        return None


# Verify is_allowed requires exact match; "attacker|allow@email.com" is rejected
def test_is_allowed_requires_exact_match() -> None:
    channel = _DummyChannel(SimpleNamespace(allow_from=["allow@email.com"]), MessageBus())

    assert channel.is_allowed("allow@email.com") is True
    assert channel.is_allowed("attacker|allow@email.com") is False
