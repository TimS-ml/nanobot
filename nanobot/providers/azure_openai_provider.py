"""Azure OpenAI provider implementation with API version 2024-10-21."""

from __future__ import annotations

import uuid
from typing import Any
from urllib.parse import urljoin

import httpx
import json_repair

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

# Azure OpenAI accepts a narrower set of message keys than some other providers
_AZURE_MSG_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name"})


class AzureOpenAIProvider(LLMProvider):
    """
    Azure OpenAI provider with API version 2024-10-21 compliance.
    
    Features:
    - Hardcoded API version 2024-10-21
    - Uses model field as Azure deployment name in URL path
    - Uses api-key header instead of Authorization Bearer
    - Uses max_completion_tokens instead of max_tokens
    - Direct HTTP calls, bypasses LiteLLM
    """

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "",
        default_model: str = "gpt-5.2-chat",
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        # Pinned API version for stable behavior across Azure updates
        self.api_version = "2024-10-21"
        
        # Both api_key and api_base are mandatory for Azure (no discovery/defaults)
        if not api_key:
            raise ValueError("Azure OpenAI api_key is required")
        if not api_base:
            raise ValueError("Azure OpenAI api_base is required")
        
        # Ensure trailing slash so urljoin resolves paths correctly
        if not api_base.endswith('/'):
            api_base += '/'
        self.api_base = api_base

    def _build_chat_url(self, deployment_name: str) -> str:
        """Build the Azure OpenAI chat completions URL.
        
        Azure requires the deployment name embedded in the URL path, unlike standard
        OpenAI which uses a "model" field in the request body.
        Format: {base}/openai/deployments/{deployment}/chat/completions?api-version={version}
        """
        base_url = self.api_base
        if not base_url.endswith('/'):
            base_url += '/'
        
        # urljoin merges base URL with deployment-specific path
        url = urljoin(
            base_url, 
            f"openai/deployments/{deployment_name}/chat/completions"
        )
        # api-version is required as a query parameter (not a header) for Azure
        return f"{url}?api-version={self.api_version}"

    def _build_headers(self) -> dict[str, str]:
        """Build headers for Azure OpenAI API.
        
        Azure uses "api-key" header for auth instead of the standard
        "Authorization: Bearer" header used by OpenAI and most other providers.
        """
        return {
            "Content-Type": "application/json",
            "api-key": self.api_key,  # Azure-specific auth header (NOT "Authorization: Bearer")
            "x-session-affinity": uuid.uuid4().hex,  # Pin to same backend for cache locality
        }

    @staticmethod
    def _supports_temperature(
        deployment_name: str,
        reasoning_effort: str | None = None,
    ) -> bool:
        """Return True when temperature is likely supported for this deployment.
        
        Reasoning models (o1, o3, o4, gpt-5) and requests with reasoning_effort
        don't support temperature — sending it would cause a 400 error.
        """
        if reasoning_effort:
            return False
        name = deployment_name.lower()
        return not any(token in name for token in ("gpt-5", "o1", "o3", "o4"))

    def _prepare_request_payload(
        self,
        deployment_name: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Prepare the request payload with Azure OpenAI 2024-10-21 compliance."""
        payload: dict[str, Any] = {
            # Sanitize messages to only include Azure-safe keys
            "messages": self._sanitize_request_messages(
                self._sanitize_empty_content(messages),
                _AZURE_MSG_KEYS,
            ),
            # Azure API 2024-10-21 uses max_completion_tokens instead of max_tokens
            # (max_tokens is deprecated/rejected in newer API versions)
            "max_completion_tokens": max(1, max_tokens),
        }

        # Only include temperature for models that support it
        if self._supports_temperature(deployment_name, reasoning_effort):
            payload["temperature"] = temperature

        # Reasoning effort controls how much "thinking" the model does (o1/o3/gpt-5)
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        return payload

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request to Azure OpenAI.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (used as deployment name).
            max_tokens: Maximum tokens in response (mapped to max_completion_tokens).
            temperature: Sampling temperature.
            reasoning_effort: Optional reasoning effort parameter.

        Returns:
            LLMResponse with content and/or tool calls.
        """
        # Azure uses the model name as the deployment name in the URL path
        deployment_name = model or self.default_model
        url = self._build_chat_url(deployment_name)
        headers = self._build_headers()
        payload = self._prepare_request_payload(
            deployment_name, messages, tools, max_tokens, temperature, reasoning_effort
        )

        try:
            # Direct HTTP call (not using openai SDK or LiteLLM) for full Azure control
            async with httpx.AsyncClient(timeout=60.0, verify=True) as client:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    return LLMResponse(
                        content=f"Azure OpenAI API Error {response.status_code}: {response.text}",
                        finish_reason="error",
                    )
                
                response_data = response.json()
                return self._parse_response(response_data)

        except Exception as e:
            return LLMResponse(
                content=f"Error calling Azure OpenAI: {repr(e)}",
                finish_reason="error",
            )

    def _parse_response(self, response: dict[str, Any]) -> LLMResponse:
        """Parse Azure OpenAI JSON response dict into our standard LLMResponse format."""
        try:
            choice = response["choices"][0]
            message = choice["message"]

            # Extract and parse tool calls if present
            tool_calls = []
            if message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    # Parse arguments from JSON string (use json_repair for malformed JSON)
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        args = json_repair.loads(args)

                    tool_calls.append(
                        ToolCallRequest(
                            id=tc["id"],
                            name=tc["function"]["name"],
                            arguments=args,
                        )
                    )

            # Extract token usage statistics
            usage = {}
            if response.get("usage"):
                usage_data = response["usage"]
                usage = {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                }

            # reasoning_content is returned by Azure's reasoning models (o1, o3, etc.)
            reasoning_content = message.get("reasoning_content") or None

            return LLMResponse(
                content=message.get("content"),
                tool_calls=tool_calls,
                finish_reason=choice.get("finish_reason", "stop"),
                usage=usage,
                reasoning_content=reasoning_content,
            )

        except (KeyError, IndexError) as e:
            # Gracefully handle unexpected response structure
            return LLMResponse(
                content=f"Error parsing Azure OpenAI response: {str(e)}",
                finish_reason="error",
            )

    def get_default_model(self) -> str:
        """Get the default model (also used as default deployment name)."""
        return self.default_model