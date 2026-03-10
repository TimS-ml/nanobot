"""OpenAI Codex Responses Provider."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, AsyncGenerator

import httpx
from loguru import logger
from oauth_cli_kit import get_token as get_codex_token

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

# Codex uses ChatGPT's backend API, not the standard OpenAI API
DEFAULT_CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
# Identifies nanobot as the originator in Codex API requests
DEFAULT_ORIGINATOR = "nanobot"


class OpenAICodexProvider(LLMProvider):
    """Use Codex OAuth to call the Responses API.
    
    Unlike other providers, Codex uses OAuth authentication (via browser login)
    instead of API keys. It uses the Responses API format, not Chat Completions.
    """

    def __init__(self, default_model: str = "openai-codex/gpt-5.1-codex"):
        # No api_key or api_base — authentication is handled via OAuth tokens
        super().__init__(api_key=None, api_base=None)
        self.default_model = default_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        model = model or self.default_model
        # Convert Chat Completions message format to Responses API input format
        system_prompt, input_items = _convert_messages(messages)

        # Obtain OAuth token via browser-based login flow (runs in thread to avoid blocking)
        token = await asyncio.to_thread(get_codex_token)
        headers = _build_headers(token.account_id, token.access)

        # Build Responses API request body (different structure from Chat Completions)
        body: dict[str, Any] = {
            "model": _strip_model_prefix(model),
            "store": False,  # Don't persist this conversation on the server
            "stream": True,  # Always stream via SSE for incremental response handling
            "instructions": system_prompt,  # System prompt goes in dedicated field
            "input": input_items,  # Converted message history in Responses API format
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],  # Request encrypted reasoning traces
            "prompt_cache_key": _prompt_cache_key(messages),  # Deterministic key for server-side caching
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }

        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}

        if tools:
            # Convert OpenAI function-calling tools to Codex's flat format
            body["tools"] = _convert_tools(tools)

        url = DEFAULT_CODEX_URL

        try:
            # First attempt with SSL verification enabled
            try:
                content, tool_calls, finish_reason = await _request_codex(url, headers, body, verify=True)
            except Exception as e:
                # Fall back to unverified SSL for corporate proxies / cert issues
                if "CERTIFICATE_VERIFY_FAILED" not in str(e):
                    raise
                logger.warning("SSL certificate verification failed for Codex API; retrying with verify=False")
                content, tool_calls, finish_reason = await _request_codex(url, headers, body, verify=False)
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        except Exception as e:
            return LLMResponse(
                content=f"Error calling Codex: {str(e)}",
                finish_reason="error",
            )

    def get_default_model(self) -> str:
        return self.default_model


def _strip_model_prefix(model: str) -> str:
    """Remove the 'openai-codex/' or 'openai_codex/' routing prefix from model names."""
    if model.startswith("openai-codex/") or model.startswith("openai_codex/"):
        return model.split("/", 1)[1]
    return model


def _build_headers(account_id: str, token: str) -> dict[str, str]:
    """Build headers for Codex Responses API with OAuth Bearer token auth."""
    return {
        "Authorization": f"Bearer {token}",  # OAuth access token (not an API key)
        "chatgpt-account-id": account_id,  # Required by ChatGPT backend to identify the account
        "OpenAI-Beta": "responses=experimental",  # Enable Responses API (still in beta)
        "originator": DEFAULT_ORIGINATOR,  # Identifies the client application
        "User-Agent": "nanobot (python)",
        "accept": "text/event-stream",  # Request SSE streaming format
        "content-type": "application/json",
    }


async def _request_codex(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    verify: bool,
) -> tuple[str, list[ToolCallRequest], str]:
    """Send a streaming POST to the Codex API and consume the SSE response."""
    async with httpx.AsyncClient(timeout=60.0, verify=verify) as client:
        # Use streaming POST because Codex always returns SSE event streams
        async with client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code != 200:
                text = await response.aread()
                raise RuntimeError(_friendly_error(response.status_code, text.decode("utf-8", "ignore")))
            return await _consume_sse(response)


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI function-calling schema to Codex flat format.
    
    OpenAI Chat Completions nests function info under {"type": "function", "function": {...}},
    while Codex Responses API uses a flat structure with name/description/parameters at top level.
    """
    converted: list[dict[str, Any]] = []
    for tool in tools:
        # Extract function definition from nested or flat format
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        converted.append({
            "type": "function",
            "name": name,
            "description": fn.get("description") or "",
            "parameters": params if isinstance(params, dict) else {},
        })
    return converted


def _convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Convert Chat Completions message list to Responses API input format.
    
    Returns (system_prompt, input_items) where system_prompt is extracted separately
    (Responses API has a dedicated 'instructions' field) and input_items is the
    conversation history in Responses API format.
    """
    system_prompt = ""
    input_items: list[dict[str, Any]] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        # System messages become the top-level 'instructions' field
        if role == "system":
            system_prompt = content if isinstance(content, str) else ""
            continue

        # User messages: convert text/image content blocks to input_text/input_image
        if role == "user":
            input_items.append(_convert_user_message(content))
            continue

        if role == "assistant":
            # Convert assistant text to Responses API "message" item with output_text block
            if isinstance(content, str) and content:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                        "status": "completed",
                        "id": f"msg_{idx}",
                    }
                )
            # Convert tool_calls to Responses API "function_call" items.
            # Each tool call becomes a separate input item (not nested in the message).
            for tool_call in msg.get("tool_calls", []) or []:
                fn = tool_call.get("function") or {}
                # Tool call IDs may encode both call_id and item_id separated by "|"
                call_id, item_id = _split_tool_call_id(tool_call.get("id"))
                call_id = call_id or f"call_{idx}"
                item_id = item_id or f"fc_{idx}"
                input_items.append(
                    {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
            continue

        # Tool result messages: convert to "function_call_output" format
        if role == "tool":
            call_id, _ = _split_tool_call_id(msg.get("tool_call_id"))
            output_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_text,
                }
            )
            continue

    return system_prompt, input_items


def _convert_user_message(content: Any) -> dict[str, Any]:
    """Convert a user message's content to Responses API input format.
    
    Maps Chat Completions content types to Responses API equivalents:
    - "text" → "input_text"
    - "image_url" → "input_image"
    """
    if isinstance(content, str):
        return {"role": "user", "content": [{"type": "input_text", "text": content}]}
    if isinstance(content, list):
        # Convert multimodal content blocks (text + images) to Responses API types
        converted: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                converted.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url")
                if url:
                    converted.append({"type": "input_image", "image_url": url, "detail": "auto"})
        if converted:
            return {"role": "user", "content": converted}
    # Fallback: empty text block if content is None or unrecognized
    return {"role": "user", "content": [{"type": "input_text", "text": ""}]}


def _split_tool_call_id(tool_call_id: Any) -> tuple[str, str | None]:
    """Split a compound tool_call_id into (call_id, item_id).
    
    Tool call IDs from Codex are stored as "call_id|item_id" to preserve
    both identifiers through the round-trip. This reverses that encoding.
    """
    if isinstance(tool_call_id, str) and tool_call_id:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, item_id or None
        return tool_call_id, None
    return "call_0", None  # Fallback ID when no tool_call_id is present


def _prompt_cache_key(messages: list[dict[str, Any]]) -> str:
    """Generate a deterministic cache key from message content for server-side caching."""
    raw = json.dumps(messages, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _iter_sse(response: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
    """Parse Server-Sent Events (SSE) stream into JSON event objects.
    
    SSE format: events are separated by blank lines, data lines start with "data:".
    The special "[DONE]" sentinel signals end of stream.
    """
    buffer: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            # Blank line = end of an SSE event block
            if buffer:
                # Extract and join all "data:" lines from this event
                data_lines = [l[5:].strip() for l in buffer if l.startswith("data:")]
                buffer = []
                if not data_lines:
                    continue
                data = "\n".join(data_lines).strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    yield json.loads(data)
                except Exception:
                    continue  # Skip malformed JSON events
            continue
        buffer.append(line)


async def _consume_sse(response: httpx.Response) -> tuple[str, list[ToolCallRequest], str]:
    """Consume all SSE events from a Codex response, accumulating text and tool calls.
    
    Handles the Responses API streaming event types:
    - response.output_text.delta: incremental text content
    - response.output_item.added/done: function call lifecycle
    - response.function_call_arguments.delta/done: streamed tool arguments
    - response.completed: final status
    """
    content = ""
    tool_calls: list[ToolCallRequest] = []
    # Buffer tool call arguments as they stream in, keyed by call_id
    tool_call_buffers: dict[str, dict[str, Any]] = {}
    finish_reason = "stop"

    async for event in _iter_sse(response):
        event_type = event.get("type")
        if event_type == "response.output_item.added":
            # New function call started — initialize buffer to accumulate streamed arguments
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                if not call_id:
                    continue
                tool_call_buffers[call_id] = {
                    "id": item.get("id") or "fc_0",
                    "name": item.get("name"),
                    "arguments": item.get("arguments") or "",
                }
        elif event_type == "response.output_text.delta":
            # Incremental text content — append to accumulated content
            content += event.get("delta") or ""
        elif event_type == "response.function_call_arguments.delta":
            # Incremental tool call arguments — append to buffer
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] += event.get("delta") or ""
        elif event_type == "response.function_call_arguments.done":
            # Final tool call arguments — replace buffer with complete value
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] = event.get("arguments") or ""
        elif event_type == "response.output_item.done":
            # Function call completed — parse arguments and emit ToolCallRequest
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                if not call_id:
                    continue
                buf = tool_call_buffers.get(call_id) or {}
                args_raw = buf.get("arguments") or item.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw)
                except Exception:
                    args = {"raw": args_raw}  # Preserve raw args if JSON parsing fails
                # Encode both call_id and item_id in the ID for round-trip fidelity
                tool_calls.append(
                    ToolCallRequest(
                        id=f"{call_id}|{buf.get('id') or item.get('id') or 'fc_0'}",
                        name=buf.get("name") or item.get("name"),
                        arguments=args,
                    )
                )
        elif event_type == "response.completed":
            # Map Codex response status to standard finish_reason
            status = (event.get("response") or {}).get("status")
            finish_reason = _map_finish_reason(status)
        elif event_type in {"error", "response.failed"}:
            raise RuntimeError("Codex response failed")

    return content, tool_calls, finish_reason


# Map Codex response statuses to standard LLM finish_reason values
_FINISH_REASON_MAP = {"completed": "stop", "incomplete": "length", "failed": "error", "cancelled": "error"}


def _map_finish_reason(status: str | None) -> str:
    """Convert Codex response status to standard finish_reason."""
    return _FINISH_REASON_MAP.get(status or "completed", "stop")


def _friendly_error(status_code: int, raw: str) -> str:
    """Convert HTTP error codes to user-friendly error messages."""
    if status_code == 429:
        return "ChatGPT usage quota exceeded or rate limit triggered. Please try again later."
    return f"HTTP {status_code}: {raw}"
