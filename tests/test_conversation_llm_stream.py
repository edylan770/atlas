"""Tests for ConversationLLM.reply_stream."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from imagecb.models.conversation_llm import ConversationLLM


def test_reply_stream_bedrock_yields_deltas():
    llm = ConversationLLM(provider="bedrock", model="test-model")
    mock_runtime = MagicMock()
    mock_runtime.converse_stream.return_value = {
        "stream": [
            {"contentBlockDelta": {"delta": {"text": "Hello"}}},
            {"contentBlockDelta": {"delta": {"text": " world"}}},
            {"messageStop": {}},
        ]
    }
    with patch(
        "imagecb.models.bedrock_client.get_bedrock_runtime",
        return_value=mock_runtime,
    ):
        chunks = list(llm.reply_stream("payload"))
    assert chunks == ["Hello", " world"]


def test_reply_stream_openai_yields_deltas():
    llm = ConversationLLM(provider="openai", model="gpt-test")
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="Hi"))]
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock(delta=MagicMock(content=None))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = [chunk1, chunk2]

    with patch("imagecb.models.conversation_llm.get_openai_client", return_value=mock_client):
        chunks = list(llm.reply_stream("payload"))
    assert chunks == ["Hi"]
    mock_client.chat.completions.create.assert_called_once()
    assert mock_client.chat.completions.create.call_args.kwargs.get("stream") is True


def test_reply_stream_anthropic_yields_deltas():
    llm = ConversationLLM(provider="anthropic", model="claude-test")
    mock_stream = MagicMock()
    mock_stream.text_stream = ["One", " two"]
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_stream
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = mock_ctx

    with patch("imagecb.models.conversation_llm.get_anthropic_client", return_value=mock_client):
        chunks = list(llm.reply_stream("payload"))
    assert chunks == ["One", " two"]


def test_reply_uses_stream_aggregate():
    llm = ConversationLLM(provider="bedrock", model="test-model")
    with patch.object(llm, "reply_stream", return_value=iter(["a", "b"])):
        assert llm.reply("x") == "ab"
