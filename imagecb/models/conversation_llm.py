"""LLM for natural-language assistant replies after retrieval."""

from __future__ import annotations

from typing import Iterator, Optional

from imagecb.config import SETTINGS

CONVERSATION_SYSTEM_PROMPT = """You are the assistant for an image search app over \
ingested slides, PDFs, and standalone images. After each search, write a short reply in \
Markdown (1–3 short paragraphs), then an optional Use cases section:

1. Summarize in plain language what was found, or say plainly that nothing matched well.
2. Refer to assets by their generated titles from the context (the name before the match \
percent). Do not quote source filenames unless the user asks for them.
3. End with a short **Use cases** section listing distinct use cases taken only from the \
`Use case:` fields in the context. Deduplicate similar wording. Skip this section if none \
are present. Do not invent use cases.

Rules:
- Never invent titles, authors, use cases, or images not present in the context.
- If the interpretation notes say matches are weak, be honest about it.
- Keep tone friendly and concise. No JSON. No code blocks."""


class ConversationLLM:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None) -> None:
        self.provider = (provider or SETTINGS.llm_provider).lower()
        self.model = model or SETTINGS.llm_model

    def reply(self, user_payload: str) -> str:
        return "".join(self.reply_stream(user_payload)).strip()

    def reply_stream(self, user_payload: str) -> Iterator[str]:
        if self.provider == "bedrock":
            yield from self._reply_stream_bedrock(user_payload)
        elif self.provider == "openai":
            yield from self._reply_stream_openai(user_payload)
        elif self.provider == "anthropic":
            yield from self._reply_stream_anthropic(user_payload)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _reply_stream_bedrock(self, user_payload: str) -> Iterator[str]:
        from imagecb.models.bedrock_client import bedrock_converse_stream

        response = bedrock_converse_stream(
            modelId=self.model,
            system=[{"text": CONVERSATION_SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_payload}]}],
            inferenceConfig={"temperature": 0.4, "maxTokens": 1000},
        )
        for event in response.get("stream", []):
            delta = event.get("contentBlockDelta")
            if not delta:
                continue
            text = delta.get("delta", {}).get("text")
            if text:
                yield text

    def _reply_stream_openai(self, user_payload: str) -> Iterator[str]:
        from openai import OpenAI

        client = OpenAI(api_key=SETTINGS.openai_api_key)
        stream = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CONVERSATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.4,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _reply_stream_anthropic(self, user_payload: str) -> Iterator[str]:
        import anthropic

        client = anthropic.Anthropic(api_key=SETTINGS.anthropic_api_key)
        with client.messages.stream(
            model=self.model,
            max_tokens=1000,
            system=CONVERSATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_payload}],
        ) as stream:
            yield from stream.text_stream


_llm: Optional[ConversationLLM] = None


def get_conversation_llm() -> ConversationLLM:
    global _llm
    if _llm is None:
        _llm = ConversationLLM()
    return _llm
