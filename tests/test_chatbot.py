"""Tests for src/chatbot.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.chatbot import (
    ConversationMemory,
    KisanSaathiChatbot,
    MessageRole,
    is_agriculture_related,
    tool_market_price,
    tool_nearby_store_finder,
)


def test_is_agriculture_related_true_for_farming_question():
    assert is_agriculture_related("Mere tomato ke patte pe kale daag hain") is True


def test_is_agriculture_related_true_for_english_question():
    assert is_agriculture_related("What pesticide should I use for my crop?") is True


def test_is_agriculture_related_false_for_offtopic_question():
    assert is_agriculture_related("What is the capital of France?") is False


def test_conversation_memory_respects_max_turns():
    memory = ConversationMemory(max_turns=3)
    for i in range(5):
        memory.add(MessageRole.USER, f"message {i}")
    assert len(memory) == 3


def test_conversation_memory_as_gemini_history_format():
    memory = ConversationMemory(max_turns=5)
    memory.add(MessageRole.USER, "hello")
    memory.add(MessageRole.ASSISTANT, "hi there")
    history = memory.as_gemini_history()
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "model"
    assert history[0]["parts"] == ["hello"]


def test_conversation_memory_clear():
    memory = ConversationMemory()
    memory.add(MessageRole.USER, "hello")
    memory.clear()
    assert len(memory) == 0


def test_tool_market_price_returns_mock_price():
    result = tool_market_price("tomato")
    assert result["crop"] == "tomato"
    assert isinstance(result["price_per_quintal_inr"], int)


def test_tool_nearby_store_finder_returns_stores():
    result = tool_nearby_store_finder("Nashik")
    assert result["location"] == "Nashik"
    assert len(result["stores"]) >= 1


def test_chatbot_send_message_refuses_offtopic():
    mock_rag_engine = MagicMock()
    mock_gemini_client = MagicMock()  # should never be called for off-topic queries

    bot = KisanSaathiChatbot(rag_engine=mock_rag_engine, gemini_client=mock_gemini_client)
    reply = bot.send_message("What is the capital of France?")

    assert "farming" in reply.lower() or "kheti" in reply.lower()
    mock_gemini_client.start_chat.assert_not_called()


def test_chatbot_send_message_calls_gemini_for_agriculture_question():
    mock_rag_engine = MagicMock()
    mock_chat = MagicMock()

    mock_response = MagicMock()
    mock_response.text = "Spray Mancozeb for early blight."
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [MagicMock(function_call=None)]
    mock_response.candidates = [mock_candidate]
    mock_chat.send_message.return_value = mock_response

    mock_gemini_client = MagicMock()
    mock_gemini_client.start_chat.return_value = mock_chat

    bot = KisanSaathiChatbot(rag_engine=mock_rag_engine, gemini_client=mock_gemini_client)
    reply = bot.send_message("Tomato mein kaunsi bimari hai?")

    assert reply == "Spray Mancozeb for early blight."
    assert len(bot.memory) == 2
