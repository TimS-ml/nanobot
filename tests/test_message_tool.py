# Tests for the MessageTool used by the agent to send messages to channels.

import pytest

from nanobot.agent.tools.message import MessageTool


# Verify that executing message tool without setting a target channel/chat returns an error
@pytest.mark.asyncio
async def test_message_tool_returns_error_when_no_target_context() -> None:
    tool = MessageTool()
    result = await tool.execute(content="test")
    assert result == "Error: No target channel/chat specified"
