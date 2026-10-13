"""Thin wrapper around the ``google-genai`` SDK.

The original ``google-generativeai`` package (import path
``google.generativeai``) is fully deprecated — Google's own package now
raises a ``FutureWarning`` on import and receives no further updates or bug
fixes. This module isolates the migration to the current ``google-genai``
package (import path ``google.genai``) behind a small, stable interface
(``generate_content`` / ``start_chat`` / ``send_message``) so the rest of
the codebase — and its tests, which inject mock clients matching this same
shape — doesn't need to know which SDK generation is underneath.
"""

from __future__ import annotations

from src.config import GEMINI_API_KEY, configure_logging, require_api_key

logger = configure_logging(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


class GeminiClientError(Exception):
    """Raised when the Gemini client cannot be constructed."""


class GeminiChatSession:
    """Adapts a ``google.genai`` ``Chat`` to the ``.send_message()`` shape
    the rest of the codebase (and its tests) expect."""

    def __init__(self, chat) -> None:
        self._chat = chat

    def send_message(self, message):
        return self._chat.send_message(message)


class GeminiTextClient:
    """Minimal client for one-shot prompts: ``.generate_content(prompt)``."""

    def __init__(self, model_name: str = DEFAULT_GEMINI_MODEL, api_key: str | None = None) -> None:
        from google import genai

        try:
            key = require_api_key("GEMINI_API_KEY", api_key or GEMINI_API_KEY)
        except Exception as exc:  # noqa: BLE001 - normalize to GeminiClientError
            raise GeminiClientError(str(exc)) from exc

        self._client = genai.Client(api_key=key)
        self._model_name = model_name

    def generate_content(self, prompt: str):
        return self._client.models.generate_content(model=self._model_name, contents=prompt)


class GeminiChatClient:
    """Minimal client for multi-turn, tool-using chat sessions.

    Exposes ``.start_chat(history=...) -> GeminiChatSession`` matching the
    interface the previous (deprecated) SDK generation provided, so
    ``src.chatbot.KisanSaathiChatbot`` doesn't need to change when the
    underlying SDK does.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
        system_instruction: str | None = None,
        tool_declarations: list[dict] | None = None,
    ) -> None:
        from google import genai
        from google.genai import types

        try:
            key = require_api_key("GEMINI_API_KEY", api_key or GEMINI_API_KEY)
        except Exception as exc:  # noqa: BLE001 - normalize to GeminiClientError
            raise GeminiClientError(str(exc)) from exc

        self._client = genai.Client(api_key=key)
        self._model_name = model_name

        tools = None
        if tool_declarations:
            tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(**decl) for decl in tool_declarations
                    ]
                )
            ]

        self._config = None
        if system_instruction or tools:
            self._config = types.GenerateContentConfig(
                system_instruction=system_instruction, tools=tools
            )

    def start_chat(self, history: list[dict] | None = None) -> GeminiChatSession:
        chat = self._client.chats.create(
            model=self._model_name, config=self._config, history=history or []
        )
        return GeminiChatSession(chat)
