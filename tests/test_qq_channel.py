# Tests for the QQ channel: inbound group message routing and outbound
# group message sending with msg_seq tracking.

from types import SimpleNamespace

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.qq import QQChannel
from nanobot.config.schema import QQConfig


# Fake QQ API that records C2C and group message calls
class _FakeApi:
    def __init__(self) -> None:
        self.c2c_calls: list[dict] = []
        self.group_calls: list[dict] = []

    async def post_c2c_message(self, **kwargs) -> None:
        self.c2c_calls.append(kwargs)

    async def post_group_message(self, **kwargs) -> None:
        self.group_calls.append(kwargs)


class _FakeClient:
    def __init__(self) -> None:
        self.api = _FakeApi()


# Verify inbound group messages set sender_id and chat_id correctly from QQ data
@pytest.mark.asyncio
async def test_on_group_message_routes_to_group_chat_id() -> None:
    channel = QQChannel(QQConfig(app_id="app", secret="secret", allow_from=["user1"]), MessageBus())

    data = SimpleNamespace(
        id="msg1",
        content="hello",
        group_openid="group123",
        author=SimpleNamespace(member_openid="user1"),
    )

    await channel._on_message(data, is_group=True)

    msg = await channel.bus.consume_inbound()
    assert msg.sender_id == "user1"
    assert msg.chat_id == "group123"


# Verify outbound group messages use post_group_message API with auto-incrementing msg_seq
@pytest.mark.asyncio
async def test_send_group_message_uses_group_api_with_msg_seq() -> None:
    channel = QQChannel(QQConfig(app_id="app", secret="secret", allow_from=["*"]), MessageBus())
    channel._client = _FakeClient()
    channel._chat_type_cache["group123"] = "group"

    await channel.send(
        OutboundMessage(
            channel="qq",
            chat_id="group123",
            content="hello",
            metadata={"message_id": "msg1"},
        )
    )

    assert len(channel._client.api.group_calls) == 1
    call = channel._client.api.group_calls[0]
    assert call["group_openid"] == "group123"
    assert call["msg_id"] == "msg1"
    assert call["msg_seq"] == 2
    assert not channel._client.api.c2c_calls
