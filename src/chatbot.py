"""Kisan Saathi ("Farmer's Companion") chatbot.

Implementation note: the original spec called for "LangChain
ConversationChain + Gemini API". In practice, Gemini's native function
calling (via ``google-generativeai``'s ``Tool``/``FunctionDeclaration``
API) gives more reliable, structured tool invocation than routing Gemini
through LangChain's older ``ConversationChain`` abstraction, so that's
what this module uses — LangChain remains in the stack for the RAG /
ChromaDB retrieval side (``rag_engine.py``). Swap the ``_call_gemini``
internals for a LangChain agent executor if a house standard requires it.

Tools exposed to the model:
    disease_lookup        — search the knowledge base by symptom description
    treatment_retriever    — RAG lookup for a known disease or free-text question
    weather_advisory       — OpenWeatherMap-based spray/disease-risk advisory
    market_price           — mock agri commodity price lookup
    nearby_store_finder     — mock nearby agri-store lookup

Guardrails: a lightweight keyword-based domain classifier declines
off-topic requests before they reach the LLM, keeping the bot scoped to
agriculture as required by the project brief.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from src.config import configure_logging
from src.rag_engine import RAGEngine, RAGEngineError
from src.translator import DetectedLanguage, detect_language
from src.weather_service import (
    WeatherServiceError,
    build_weather_alert_message,
    fetch_five_day_forecast,
    recommend_spray_days,
)

logger = configure_logging(__name__)

MAX_CONVERSATION_TURNS = 10
GEMINI_MODEL_NAME = "gemini-3.8-flash"

# Keyword-based domain guardrail. Anything scoring zero hits is treated as
# off-topic. Deliberately broad (crops, pests, weather, farming inputs,
# government schemes) since farmers' questions legitimately span all of
# these around a core disease-detection use case.
AGRICULTURE_KEYWORDS = frozenset(
    {
        "crop", "fasal", "plant", "paudha", "disease", "bimari", "rog", "pest",
        "keet", "kirtan", "spray", "chhidkav", "pesticide", "fungicide",
        "fertilizer", "khaad", "urvarak", "soil", "mitti", "weather", "mausam",
        "rain", "baarish", "irrigation", "sinchai", "seed", "beej", "harvest",
        "katai", "farmer", "kisan", "agriculture", "krishi", "leaf", "patta",
        "patte", "tomato", "potato", "wheat", "rice", "corn", "maize", "apple",
        "grape", "pepper", "mirch", "aloo", "gehun", "chawal", "makka", "market",
        "mandi", "price", "bhav", "dukan", "store", "scheme", "yojana", "insurance",
        "bima", "organic", "jaivik", "chemical", "dawai", "dawa", "vaccine",
        "tractor", "khet", "farm",
    }
)


class ChatbotError(Exception):
    """Raised for unrecoverable chatbot errors (not off-topic refusals)."""


class OffTopicRefusal(Exception):
    """Raised internally when a query fails the agriculture-domain guardrail."""


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ConversationTurn:
    role: MessageRole
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConversationMemory:
    """Fixed-window conversation memory — keeps the last N turns only."""

    def __init__(self, max_turns: int = MAX_CONVERSATION_TURNS) -> None:
        self._turns: deque[ConversationTurn] = deque(maxlen=max_turns)

    def add(self, role: MessageRole, content: str) -> None:
        self._turns.append(ConversationTurn(role=role, content=content))

    def as_gemini_history(self) -> list[dict]:
        """Format stored turns as Gemini chat history (``role``/``parts``)."""
        role_map = {MessageRole.USER: "user", MessageRole.ASSISTANT: "model"}
        return [{"role": role_map[t.role], "parts": [t.content]} for t in self._turns]

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)


def is_agriculture_related(query: str) -> bool:
    """Guardrail: check whether a query plausibly concerns agriculture.

    Args:
        query: The user's raw message.

    Returns:
        ``True`` if at least one agriculture-domain keyword is present.
    """
    lowered = query.lower()
    return any(keyword in lowered for keyword in AGRICULTURE_KEYWORDS)


# ---------------------------------------------------------------------------
# Tool implementations. Each takes/returns plain JSON-serializable types so
# they can be wired directly into Gemini's function-calling response loop.
# ---------------------------------------------------------------------------

def tool_disease_lookup(symptom_description: str, rag_engine: RAGEngine) -> dict:
    """Search the knowledge base for diseases matching a symptom description.

    Args:
        symptom_description: Free-text symptom description from the farmer,
            e.g. "kale daag tomato ke patte pe".
        rag_engine: Shared :class:`~src.rag_engine.RAGEngine` instance.

    Returns:
        Dict with a ``matches`` list of retrieved passages.
    """
    passages = rag_engine.answer_free_text_query(symptom_description, top_k=3)
    return {"matches": passages}


def tool_treatment_retriever(disease_name_or_question: str, rag_engine: RAGEngine) -> dict:
    """Retrieve a treatment plan for a known disease, or context for a question.

    Args:
        disease_name_or_question: Either an exact ``CLASS_NAMES`` entry or a
            free-text follow-up question (e.g. "kaunsi dawai use karein?").
        rag_engine: Shared :class:`~src.rag_engine.RAGEngine` instance.

    Returns:
        Dict form of a :class:`~src.rag_engine.TreatmentPlan`, or a
        ``{"matches": [...]}`` fallback for free-text questions.
    """
    try:
        plan = rag_engine.get_treatment_plan(disease_name_or_question)
        return plan.to_dict()
    except RAGEngineError:
        passages = rag_engine.answer_free_text_query(disease_name_or_question, top_k=3)
        return {"matches": passages}


def tool_weather_advisory(city_name: str) -> dict:
    """Fetch a 5-day disease-risk weather advisory for a city.

    Args:
        city_name: City name, e.g. "Nashik,IN".

    Returns:
        Dict with the forecast, recommended spray days, and an alert message.
    """
    try:
        forecasts = fetch_five_day_forecast(city_name)
    except WeatherServiceError as exc:
        return {"error": str(exc)}

    return {
        "forecast": [f.to_dict() for f in forecasts],
        "recommended_spray_days": recommend_spray_days(forecasts),
        "alert_message": build_weather_alert_message(forecasts),
    }


def tool_market_price(crop_name: str) -> dict:
    """Mock agri commodity price lookup.

    Args:
        crop_name: Crop name, e.g. "tomato".

    Returns:
        Dict with a mock price per quintal in INR. Replace with a real
        mandi (agmarknet.gov.in) API integration before production use.
    """
    mock_prices = {
        "tomato": 1800, "potato": 1200, "onion": 1500, "wheat": 2200,
        "rice": 2000, "maize": 1900, "corn": 1900, "apple": 6000, "grape": 4500,
    }
    price = mock_prices.get(crop_name.lower(), 1500)
    return {
        "crop": crop_name, "price_per_quintal_inr": price,
        "note": "Mock data for demo purposes — integrate agmarknet.gov.in for live mandi prices.",
    }


def tool_nearby_store_finder(location: str) -> dict:
    """Mock nearby agri-store lookup.

    Args:
        location: Village/town/district name.

    Returns:
        Dict with a mock list of nearby agri-input stores. Replace with a
        real Google Places-style lookup before production use.
    """
    return {
        "location": location,
        "stores": [
            {"name": "Kisan Seva Kendra", "distance_km": 2.3, "phone": "N/A"},
            {"name": "Krishi Bhandar", "distance_km": 4.1, "phone": "N/A"},
        ],
        "note": "Mock data for demo purposes — integrate a real store directory before production use.",
    }


GEMINI_TOOL_DECLARATIONS = [
    {
        "name": "disease_lookup",
        "description": "Search the knowledge base for crop diseases matching a symptom description.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"symptom_description": {"type": "STRING"}},
            "required": ["symptom_description"],
        },
    },
    {
        "name": "treatment_retriever",
        "description": "Retrieve a treatment plan for a known disease name, or context for a follow-up question about treatment.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"disease_name_or_question": {"type": "STRING"}},
            "required": ["disease_name_or_question"],
        },
    },
    {
        "name": "weather_advisory",
        "description": "Get a 5-day weather forecast with fungal disease risk and spray-day recommendations for a city.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"city_name": {"type": "STRING"}},
            "required": ["city_name"],
        },
    },
    {
        "name": "market_price",
        "description": "Get the current mandi (market) price for a crop.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"crop_name": {"type": "STRING"}},
            "required": ["crop_name"],
        },
    },
    {
        "name": "nearby_store_finder",
        "description": "Find nearby agricultural input stores for a location.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"location": {"type": "STRING"}},
            "required": ["location"],
        },
    },
]

SYSTEM_PROMPT = (
    "You are Kisan Saathi, a friendly agricultural assistant chatbot for "
    "Indian farmers, part of the AgriSense AI platform. You ONLY discuss "
    "agriculture: crop diseases, treatments, weather, market prices, and "
    "farming practices. If asked about anything else, politely say you can "
    "only help with farming questions. Detect whether the farmer is writing "
    "in Hindi, English, or Hinglish and reply in the same style. Keep "
    "answers short (2-4 sentences) and practical. Use the available tools "
    "whenever a question needs current data (disease treatment details, "
    "weather, prices, store locations) rather than guessing."
)


class KisanSaathiChatbot:
    """Stateful chatbot session wrapping Gemini function calling."""

    def __init__(self, rag_engine: RAGEngine, gemini_client=None) -> None:
        """Initialize a chatbot session.

        Args:
            rag_engine: Shared :class:`~src.rag_engine.RAGEngine` instance
                used by the disease/treatment tools.
            gemini_client: Optional pre-configured model instance (for
                testing); defaults to a fresh model with tools attached,
                using ``GEMINI_API_KEY`` from the environment.
        """
        self.rag_engine = rag_engine
        self.memory = ConversationMemory()
        self._gemini_client = gemini_client or self._build_default_client()

        self._tool_dispatch = {
            "disease_lookup": lambda args: tool_disease_lookup(args["symptom_description"], self.rag_engine),
            "treatment_retriever": lambda args: tool_treatment_retriever(
                args["disease_name_or_question"], self.rag_engine
            ),
            "weather_advisory": lambda args: tool_weather_advisory(args["city_name"]),
            "market_price": lambda args: tool_market_price(args["crop_name"]),
            "nearby_store_finder": lambda args: tool_nearby_store_finder(args["location"]),
        }

    @staticmethod
    def _build_default_client():
        from src.llm_client import GeminiChatClient, GeminiClientError

        try:
            return GeminiChatClient(
                model_name=GEMINI_MODEL_NAME,
                system_instruction=SYSTEM_PROMPT,
                tool_declarations=GEMINI_TOOL_DECLARATIONS,
            )
        except GeminiClientError as exc:
            raise ChatbotError(str(exc)) from exc

    def _execute_tool_call(self, function_call) -> dict:
        """Dispatch a Gemini function call to its Python implementation.

        Args:
            function_call: A Gemini ``FunctionCall`` object with ``.name``
                and ``.args``.

        Returns:
            JSON-serializable dict result from the tool.
        """
        tool_name = function_call.name
        args = dict(function_call.args)
        handler = self._tool_dispatch.get(tool_name)

        if handler is None:
            logger.warning("Unknown tool requested by model: %s", tool_name)
            return {"error": f"Unknown tool '{tool_name}'"}

        try:
            return handler(args)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as an error
            logger.error("Tool '%s' failed: %s", tool_name, exc)
            return {"error": str(exc)}

    def send_message(self, user_message: str) -> str:
        """Send a farmer message through the chatbot and get a reply.

        Handles the agriculture-domain guardrail, language detection,
        Gemini function-calling loop (single round of tool calls), and
        conversation memory bookkeeping.

        Args:
            user_message: The farmer's raw message text.

        Returns:
            The chatbot's reply text.
        """
        if not is_agriculture_related(user_message):
            refusal = self._build_off_topic_refusal(user_message)
            self.memory.add(MessageRole.USER, user_message)
            self.memory.add(MessageRole.ASSISTANT, refusal)
            return refusal

        detection = detect_language(user_message)
        logger.info("User message language detected: %s", detection.language.value)

        chat = self._gemini_client.start_chat(history=self.memory.as_gemini_history())

        try:
            response = chat.send_message(user_message)
        except Exception as exc:  # noqa: BLE001
            raise ChatbotError(f"Gemini chat call failed: {exc}") from exc

        # Handle a (single round of) function call(s) if the model requested one.
        candidate = response.candidates[0]
        function_calls = [
            part.function_call
            for part in candidate.content.parts
            if getattr(part, "function_call", None)
        ]

        if function_calls:
            tool_responses = []
            for fc in function_calls:
                result = self._execute_tool_call(fc)
                tool_responses.append(
                    {"function_response": {"name": fc.name, "response": result}}
                )
            response = chat.send_message(tool_responses)

        reply_text = (response.text or "").strip()
        if not reply_text:
            reply_text = (
                "Maaf kijiye, main abhi jawab nahi de saka. Kripya dobara try karein."
            )

        self.memory.add(MessageRole.USER, user_message)
        self.memory.add(MessageRole.ASSISTANT, reply_text)
        return reply_text

    @staticmethod
    def _build_off_topic_refusal(user_message: str) -> str:
        """Build a polite, scope-reminding refusal for non-agriculture queries.

        Args:
            user_message: The original message (used only for language detection).

        Returns:
            A short refusal message in the detected language style.
        """
        detection = detect_language(user_message)
        if detection.language == DetectedLanguage.ENGLISH:
            return (
                "I'm Kisan Saathi, and I can only help with farming questions — "
                "crop diseases, treatments, weather, and market prices. "
                "Please ask me something related to your farm!"
            )
        return (
            "Main Kisan Saathi hoon, aur sirf kheti se jude sawalon mein madad "
            "kar sakta hoon — fasal ki bimari, upchar, mausam, ya mandi bhav. "
            "Kripya apni kheti se related sawal poochhein!"
        )
