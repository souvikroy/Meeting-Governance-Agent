"""All generative LLM calls go through OpenRouter (OpenAI-compatible API).

`structured()` requests JSON, validates it against a Pydantic model, and does one
repair retry on malformed output. A refusal / empty / un-repairable response
returns None — the caller treats that as "unsure -> contain", never as COMMIT."""
from __future__ import annotations

import json
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from . import config

T = TypeVar("T", bound=BaseModel)


class LLM:
    def __init__(self, model: Optional[str] = None):
        config.require("OPENROUTER_API_KEY")
        from openai import OpenAI  # lazy import

        self.model = model or config.OPENROUTER_MODEL
        self._client = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://localhost/meeting-governance-agent",
                "X-Title": "Meeting Governance Agent",
            },
        )

    def _chat(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        choice = resp.choices[0]
        # Some providers signal refusals via finish_reason or an empty message.
        if getattr(choice.message, "refusal", None):
            return ""
        return choice.message.content or ""

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        text = text.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Salvage the first {...} block if the model wrapped it in prose.
            start, end = text.find("{"), text.rfind("}")
            if 0 <= start < end:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None

    def structured(self, system: str, user: str, schema: Type[T]) -> Optional[T]:
        """Return a validated `schema` instance, or None if the model refused /
        produced output that cannot be parsed or repaired."""
        raw = self._chat(system, user)
        data = self._extract_json(raw)
        if data is not None:
            try:
                return schema.model_validate(data)
            except ValidationError as e:
                # One repair attempt: show the model its own output + the error.
                repair = (
                    f"Your previous JSON was invalid: {e}\n\n"
                    f"Previous output:\n{raw}\n\n"
                    "Return corrected JSON that conforms to the required schema. "
                    "Output ONLY the JSON object."
                )
                raw2 = self._chat(system, repair)
                data2 = self._extract_json(raw2)
                if data2 is not None:
                    try:
                        return schema.model_validate(data2)
                    except ValidationError:
                        return None
        return None
