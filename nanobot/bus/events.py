"""Event types for the message bus."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# Inbound and Outbound are the only message types flowing through the MessageBus.
# They are intentionally simple dataclasses — channel-specific details go in metadata.

@dataclass
class InboundMessage:
    """Message received from a chat channel."""

    channel: str  # telegram, discord, slack, whatsapp
    sender_id: str  # User identifier
    chat_id: str  # Chat/channel identifier
    content: str  # Message text
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)  # Media URLs
    metadata: dict[str, Any] = field(default_factory=dict)  # Channel-specific data
    # Allows thread-scoped sessions (e.g. Slack threads) to override the default key
    session_key_override: str | None = None  # Optional override for thread-scoped sessions

    @property
    def session_key(self) -> str:
        """Unique key for session identification."""
        # Default format: "channel:chat_id" (e.g. "telegram:12345")
        return self.session_key_override or f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """Message to send to a chat channel."""

    channel: str  # Target channel name — must match a registered channel
    chat_id: str  # Target chat/conversation to deliver to
    content: str
    reply_to: str | None = None  # Platform message ID to reply/quote
    media: list[str] = field(default_factory=list)
    # _progress and _tool_hint metadata flags control streaming behavior in the dispatcher
    metadata: dict[str, Any] = field(default_factory=dict)


