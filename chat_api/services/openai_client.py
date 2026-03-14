from __future__ import annotations

import asyncio
from dataclasses import dataclass
import random
import re
from typing import Iterable

import httpx

from chat_api.config import Settings
from chat_api.models import MessageRecord
from chat_api.services.provider_pool import ProviderPool

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_ROLE_INSTRUCTIONS = {
    "direct_answer": "Give the direct answer first, then one decisive reason and one concrete next move.",
    "comparative_feasibility": "Compare options side-by-side and pick one winner with a short tradeoff statement.",
    "failure_modes": "List the top failure mode, why it fails, and one mitigation.",
    "implementation_sequence": "Give an implementation sequence with explicit phases and measurable gates.",
    "skeptical_counterexample": "Challenge the main assumption and show what evidence would falsify it.",
}


@dataclass
class OpenAIContextResult:
    ok: bool
    context: str = ""
    error: str | None = None
    attempts: int = 1


@dataclass
class OpenAIAnswerResult:
    ok: bool
    answer: str = ""
    error: str | None = None
    provider: str = "openai"
    role: str = ""
    attempts: int = 1


class OpenAIClient:
    def __init__(self, settings: Settings, provider_pool: ProviderPool | None = None) -> None:
        self.settings = settings
        self._last_error: str | None = None
        self._provider = "openai"
        self._provider_pool = provider_pool or ProviderPool()
        self._provider_pool.configure(self._provider, self._api_keys())

    @property
    def configured(self) -> bool:
        return self._provider_pool.configured(self._provider)

    @property
    def available(self) -> bool:
        return self.available_key_count > 0

    @property
    def key_count(self) -> int:
        return self._provider_pool.key_count(self._provider)

    @property
    def available_key_count(self) -> int:
        return self._provider_pool.available_key_count(self._provider)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def health(self) -> dict[str, object]:
        return self._provider_pool.provider_health(self._provider)

    async def extract_context(
        self,
        *,
        user_text: str,
        summary: str,
        recent_messages: Iterable[MessageRecord],
        focus: str = "",
    ) -> OpenAIContextResult:
        if not self.configured:
            return OpenAIContextResult(ok=False, error="OpenAI API key is not configured.")

        prompt = self._build_context_prompt(
            user_text=user_text,
            summary=summary,
            recent_messages=list(recent_messages),
            focus=focus,
        )
        text, err = await self._request_once(prompt)
        if err:
            self._last_error = err
            return OpenAIContextResult(ok=False, error=err, attempts=1)
        assert text is not None

        cleaned = self._sanitize_context_text(text, user_text=user_text)
        if not cleaned:
            self._last_error = "empty_context"
            return OpenAIContextResult(ok=False, error="OpenAI returned empty context.", attempts=1)

        self._last_error = None
        return OpenAIContextResult(ok=True, context=cleaned, attempts=1)

    async def answer_candidate(
        self,
        *,
        user_text: str,
        summary: str,
        recent_messages: Iterable[MessageRecord],
        role: str,
        external_context: str = "",
    ) -> OpenAIAnswerResult:
        if not self.configured:
            return OpenAIAnswerResult(ok=False, error="OpenAI API key is not configured.", role=role)
        prompt = self._build_candidate_prompt(
            user_text=user_text,
            summary=summary,
            recent_messages=list(recent_messages),
            role=role,
            external_context=external_context,
        )
        text, err = await self._request_once(prompt)
        if err:
            self._last_error = err
            return OpenAIAnswerResult(ok=False, error=err, role=role)
        assert text is not None
        cleaned = self._sanitize_answer_text(text, user_text=user_text)
        if not cleaned:
            return OpenAIAnswerResult(ok=False, error="empty_candidate_answer", role=role)
        return OpenAIAnswerResult(ok=True, answer=cleaned, role=role)

    async def _request_once(self, prompt: str) -> tuple[str | None, str | None]:
        base = self.settings.openai_endpoint.rstrip("/")
        url = f"{base}/chat/completions"
        body = {
            "model": self.settings.openai_model,
            "temperature": 0.2,
            "max_tokens": 260,
            "messages": [
                {
                    "role": "system",
                    "content": "You extract concise context for a local liquid model. Return plain text only.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        attempts = max(1, self.settings.openai_retries + 1)
        if not self.configured:
            return None, "OpenAI API key is not configured."
        total_attempts = attempts * max(1, self.key_count)
        last_err: str | None = None
        for attempt in range(total_attempts):
            key, key_id = self._provider_pool.lease(self._provider)
            if not key or not key_id:
                return None, "OpenAI API key is not configured."
            headers = {
                "authorization": f"Bearer {key}",
                "content-type": "application/json",
            }
            try:
                async with httpx.AsyncClient(timeout=self.settings.openai_timeout_seconds) as client:
                    resp = await client.post(url, headers=headers, json=body)
                if resp.status_code >= 400:
                    last_err = self._extract_error(resp)
                    self._provider_pool.report_failure(
                        self._provider,
                        key_id,
                        status_code=resp.status_code,
                        error=last_err,
                    )
                    if resp.status_code in {429, 500, 502, 503, 504} and attempt + 1 < total_attempts:
                        await asyncio.sleep(self._retry_delay_seconds(attempt))
                        continue
                    return None, last_err

                payload = resp.json()
                text = self._extract_text(payload)
                if text:
                    self._provider_pool.report_success(self._provider, key_id)
                    return text, None
                last_err = "OpenAI response had no text content."
                self._provider_pool.report_failure(self._provider, key_id, status_code=0, error=last_err)
            except Exception as exc:  # pragma: no cover - defensive
                err = str(exc).strip()
                if not err:
                    err = repr(exc)
                last_err = f"OpenAI transport error ({exc.__class__.__name__}): {err}"
                self._provider_pool.report_failure(self._provider, key_id, status_code=0, error=last_err)
                if attempt + 1 < total_attempts:
                    await asyncio.sleep(self._retry_delay_seconds(attempt))
                    continue

        return None, last_err or "OpenAI request failed."

    def _retry_delay_seconds(self, attempt: int) -> float:
        base = 0.35 * (2**attempt)
        return min(2.0, base + random.uniform(0.0, 0.15))

    def _extract_error(self, resp: httpx.Response) -> str:
        try:
            payload = resp.json()
            err = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(err, dict):
                msg = err.get("message")
                if isinstance(msg, str) and msg.strip():
                    return f"OpenAI HTTP {resp.status_code}: {msg.strip()}"
        except Exception:
            pass
        text = resp.text.strip()
        if len(text) > 220:
            text = text[:220].rstrip() + "..."
        return f"OpenAI HTTP {resp.status_code}: {text}"

    def _extract_text(self, payload: dict) -> str:
        choices = payload.get("choices") or []
        for choice in choices:
            msg = choice.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        return ""

    def _build_context_prompt(
        self,
        *,
        user_text: str,
        summary: str,
        recent_messages: Iterable[MessageRecord],
        focus: str,
    ) -> str:
        history_lines = []
        for msg in list(recent_messages)[-8:]:
            if msg.role not in {"user", "assistant"}:
                continue
            history_lines.append(f"{msg.role}: {msg.content}")
        history_text = "\n".join(history_lines)
        focus_text = focus.strip() or "core intent and constraints"
        return (
            "Extract concise context for a local liquid model.\n"
            "Return 3-5 short factual lines (not bullets), each <= 16 words.\n"
            "No preamble, no markdown, no labels.\n"
            f"Focus: {focus_text}\n\n"
            f"User:\n{user_text}\n\n"
            f"Summary:\n{summary or '(none)'}\n\n"
            f"Recent messages:\n{history_text or '(none)'}\n"
        )

    def _build_candidate_prompt(
        self,
        *,
        user_text: str,
        summary: str,
        recent_messages: list[MessageRecord],
        role: str,
        external_context: str,
    ) -> str:
        role_instruction = _ROLE_INSTRUCTIONS.get(role, "Provide the strongest concise reasoning path.")
        history_lines = []
        for msg in recent_messages[-8:]:
            if msg.role not in {"user", "assistant"}:
                continue
            history_lines.append(f"{msg.role}: {msg.content}")
        history_text = "\n".join(history_lines)
        return (
            "You are a reasoning worker for a jury-based local model.\n"
            "Return plain text only in 3-6 sentences.\n"
            "Sentence 1 must answer the user directly.\n"
            "Do not repeat the user prompt verbatim. No markdown headings.\n"
            "Avoid meta phrases like 'based on available context' or 'user asks'.\n"
            f"Role objective: {role_instruction}\n"
            f"Reasoning role: {role}\n\n"
            f"User query:\n{user_text}\n\n"
            f"Context summary:\n{summary or '(none)'}\n\n"
            f"External context:\n{external_context or '(none)'}\n\n"
            f"Recent turns:\n{history_text or '(none)'}\n"
        )

    def _sanitize_context_text(self, text: str, *, user_text: str) -> str:
        raw = text.strip()
        if not raw:
            return ""
        lines = [line.strip() for line in re.split(r"[\n;]+", raw) if line.strip()]
        out: list[str] = []
        seen: set[str] = set()
        user_tokens = self._tokens(user_text)
        for line in lines:
            text_line = " ".join(line.split())
            text_line = re.sub(r"^[*\-\d\.\)\(]+\s*", "", text_line).strip()
            if not text_line:
                continue
            if self._is_meta_sentence(text_line):
                continue
            if self._looks_like_echo(text_line, user_tokens):
                continue
            norm = " ".join(w.lower() for w in _WORD_RE.findall(text_line))
            if not norm:
                continue
            if len(norm.split()) < 3:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            if text_line[-1] not in ".!?":
                text_line += "."
            out.append(text_line)
            if len(out) >= 10:
                break
        return " ".join(out)

    def _sanitize_answer_text(self, text: str, *, user_text: str) -> str:
        cleaned = " ".join(text.replace("\r\n", "\n").split())
        if not cleaned:
            return ""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if s.strip()]
        user_tokens = self._tokens(user_text)
        kept: list[str] = []
        for sentence in sentences:
            sentence = re.sub(r"(?i)^based on (the )?available context,?\s*", "", sentence).strip()
            sentence = re.sub(r"(?i)^user (asks|asked|proposed|wants|demands)\s*[:,-]\s*", "", sentence).strip()
            if not sentence:
                continue
            if self._is_meta_sentence(sentence):
                continue
            if self._looks_like_echo(sentence, user_tokens):
                continue
            kept.append(sentence)
            if len(kept) >= 6:
                break
        out = " ".join(kept).strip()
        if out and out[-1] not in ".!?":
            out += "."
        return out

    def _tokens(self, text: str) -> set[str]:
        return {t for t in _WORD_RE.findall(text.lower()) if t}

    def _looks_like_echo(self, sentence: str, user_tokens: set[str]) -> bool:
        if not user_tokens:
            return False
        sent_tokens = self._tokens(sentence)
        if not sent_tokens:
            return False
        overlap = len(sent_tokens & user_tokens) / max(1, len(sent_tokens | user_tokens))
        return overlap >= 0.74

    def _is_meta_sentence(self, sentence: str) -> bool:
        low = " ".join(sentence.lower().split())
        prefixes = (
            "based on the available context",
            "user asks",
            "the user asks",
            "user proposed",
            "user wants",
            "user demands",
            "user query",
            "previous user",
            "previous assistant",
            "working summary",
            "active topic terms",
        )
        return any(low.startswith(prefix) for prefix in prefixes)

    def _api_keys(self) -> list[str]:
        keys = [k.strip() for k in self.settings.openai_api_keys if k.strip()]
        if keys:
            return keys
        primary = self.settings.openai_api_key.strip()
        return [primary] if primary else []
