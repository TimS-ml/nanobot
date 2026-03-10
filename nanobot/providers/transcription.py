"""Voice transcription provider using Groq."""

import os
from pathlib import Path

import httpx
from loguru import logger


class GroqTranscriptionProvider:
    """
    Voice transcription provider using Groq's Whisper API.

    Groq offers extremely fast transcription with a generous free tier.
    """

    def __init__(self, api_key: str | None = None):
        # Fall back to env var if no key provided directly
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        # Groq exposes an OpenAI-compatible transcription endpoint
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using Groq.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text.
        """
        if not self.api_key:
            logger.warning("Groq API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    # Multipart form upload: audio file + model selection
                    # whisper-large-v3 is Groq's fastest and most accurate Whisper variant
                    files = {
                        "file": (path.name, f),
                        "model": (None, "whisper-large-v3"),
                    }
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                    }

                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        files=files,
                        timeout=60.0  # Transcription can take a while for long audio
                    )

                    response.raise_for_status()
                    data = response.json()
                    # Groq returns {"text": "transcribed content"} in OpenAI format
                    return data.get("text", "")

        except Exception as e:
            logger.error("Groq transcription error: {}", e)
            return ""
