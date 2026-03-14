from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import random
import re
from typing import Iterable

import httpx

from chat_api.config import Settings
from chat_api.models import MessageRecord
from chat_api.services.provider_pool import ProviderPool

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENTENCE_RE = re.compile(r"[.!?](?:\s|$)")
_REPEATED_PHRASE_RE = re.compile(r"\b(.{8,}?)\b(?:\s+\1\b){1,}", re.IGNORECASE)
_ROLE_INSTRUCTIONS = {
    "direct_answer": "Give the direct answer first, then one decisive reason and one concrete next move.",
    "comparative_feasibility": "Compare options side-by-side and pick one winner with a short tradeoff statement.",
    "failure_modes": "List the top failure mode, why it fails, and one mitigation.",
    "implementation_sequence": "Give an implementation sequence with explicit phases and measurable gates.",
    "skeptical_counterexample": "Challenge the main assumption and show what evidence would falsify it.",
}


@dataclass
class OutputQuality:
    word_count: int
    sentence_count: int
    ends_cleanly: bool
    has_repetition: bool
    is_valid: bool
    flags: list[str] = field(default_factory=list)


@dataclass
class GeminiResult:
    ok: bool
    text: str = ""
    error: str | None = None
    attempts: int = 1
    quality_flags: list[str] = field(default_factory=list)


@dataclass
class GeminiContextResult:
    ok: bool
    context: str = ""
    error: str | None = None
    attempts: int = 1


@dataclass
class GeminiAnswerResult:
    ok: bool
    answer: str = ""
    error: str | None = None
    provider: str = "gemini"
    role: str = ""
    attempts: int = 1


class GeminiClient:
    def __init__(self, settings: Settings, provider_pool: ProviderPool | None = None) -> None:
        self.settings = settings
        self._last_error: str | None = None
        self._provider = "gemini"
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

    async def generate(
        self,
        *,
        user_text: str,
        deterministic_draft: str,
        summary: str,
        recent_messages: Iterable[MessageRecord],
    ) -> GeminiResult:
        if not self.configured:
            return GeminiResult(ok=False, error="Gemini API key is not configured.")

        history = list(recent_messages)
        first_prompt = self._build_prompt(
            user_text=user_text,
            deterministic_draft=deterministic_draft,
            summary=summary,
            recent_messages=history,
            repair_mode=False,
            prior_output=None,
        )
        first_text, first_err = await self._request_once(first_prompt)
        if first_err:
            self._last_error = first_err
            return GeminiResult(ok=False, error=first_err, attempts=1)

        assert first_text is not None
        cleaned_first = self._sanitize_text(first_text)
        quality_first = self.evaluate_output(cleaned_first)

        if not self.settings.response_guard_enabled or quality_first.is_valid:
            self._last_error = None
            return GeminiResult(
                ok=True,
                text=cleaned_first,
                attempts=1,
                quality_flags=quality_first.flags,
            )

        if self.settings.response_regen_attempts <= 0:
            self._last_error = "quality_guard_failed"
            return GeminiResult(
                ok=False,
                error="quality_guard_failed",
                attempts=1,
                quality_flags=quality_first.flags,
            )

        repair_prompt = self._build_prompt(
            user_text=user_text,
            deterministic_draft=deterministic_draft,
            summary=summary,
            recent_messages=history,
            repair_mode=True,
            prior_output=cleaned_first,
        )
        second_text, second_err = await self._request_once(repair_prompt)
        if second_err:
            self._last_error = second_err
            return GeminiResult(
                ok=False,
                error=second_err,
                attempts=2,
                quality_flags=quality_first.flags,
            )

        assert second_text is not None
        cleaned_second = self._sanitize_text(second_text)
        quality_second = self.evaluate_output(cleaned_second)
        if quality_second.is_valid:
            self._last_error = None
            return GeminiResult(
                ok=True,
                text=cleaned_second,
                attempts=2,
                quality_flags=quality_second.flags,
            )

        self._last_error = "quality_guard_failed"
        return GeminiResult(
            ok=False,
            error="quality_guard_failed",
            attempts=2,
            quality_flags=sorted(set(quality_first.flags + quality_second.flags)),
        )

    async def generate_raw(
        self,
        *,
        user_text: str,
        recent_messages: Iterable[MessageRecord],
    ) -> GeminiResult:
        if not self.configured:
            return GeminiResult(ok=False, error="Gemini API key is not configured.")

        contents: list[dict] = []
        for msg in list(recent_messages)[-12:]:
            role = "model" if msg.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg.content}]})
        contents.append({"role": "user", "parts": [{"text": user_text}]})

        text, err = await self._request_once_with_contents(contents)
        if err:
            self._last_error = err
            return GeminiResult(ok=False, error=err, attempts=1)
        assert text is not None
        self._last_error = None
        return GeminiResult(ok=True, text=text.strip(), attempts=1)

    async def extract_context(
        self,
        *,
        user_text: str,
        summary: str,
        recent_messages: Iterable[MessageRecord],
        focus: str = "",
    ) -> GeminiContextResult:
        if not self.configured:
            return GeminiContextResult(ok=False, error="Gemini API key is not configured.")

        prompt = self._build_context_prompt(
            user_text=user_text,
            summary=summary,
            recent_messages=list(recent_messages),
            focus=focus,
        )
        text, err = await self._request_once(prompt)
        if err:
            self._last_error = err
            return GeminiContextResult(ok=False, error=err, attempts=1)
        assert text is not None
        cleaned = self._sanitize_context_text(text, user_text=user_text)
        if not cleaned:
            self._last_error = "empty_context"
            return GeminiContextResult(ok=False, error="Gemini returned empty context.", attempts=1)
        self._last_error = None
        return GeminiContextResult(ok=True, context=cleaned, attempts=1)

    async def answer_candidate(
        self,
        *,
        user_text: str,
        summary: str,
        recent_messages: Iterable[MessageRecord],
        role: str,
        external_context: str = "",
    ) -> GeminiAnswerResult:
        if not self.configured:
            return GeminiAnswerResult(ok=False, error="Gemini API key is not configured.", role=role)
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
            return GeminiAnswerResult(ok=False, error=err, role=role)
        assert text is not None
        cleaned = self._sanitize_candidate_answer(text, user_text=user_text)
        if not cleaned:
            return GeminiAnswerResult(ok=False, error="empty_candidate_answer", role=role)
        return GeminiAnswerResult(ok=True, answer=cleaned, role=role)

    def evaluate_output(self, text: str) -> OutputQuality:
        raw = text.strip()
        words = _WORD_RE.findall(raw)
        word_count = len(words)
        sentence_count = len(_SENTENCE_RE.findall(raw))
        ends_cleanly = bool(raw) and raw[-1] in ".!?"
        has_repetition = bool(_REPEATED_PHRASE_RE.search(raw))

        flags: list[str] = []
        if self.settings.response_incomplete_tail_guard and not ends_cleanly:
            flags.append("incomplete_tail")

        mode = self.settings.response_depth_mode
        min_words = 10
        min_sentences = 1
        max_words = 220
        if mode == "balanced":
            min_words = self.settings.response_min_words_balanced
            min_sentences = self.settings.response_min_sentences_balanced
            max_words = self.settings.response_max_words_balanced
        elif mode == "detailed":
            min_words = max(90, self.settings.response_min_words_balanced + 20)
            min_sentences = max(4, self.settings.response_min_sentences_balanced + 1)
            max_words = max(320, self.settings.response_max_words_balanced + 120)

        strict_min_words = min_words
        soft_min_words = max(28, int(min_words * 0.8))
        if word_count < soft_min_words:
            flags.append("too_short")
        elif word_count < strict_min_words:
            flags.append("under_target_words")
        if sentence_count < min_sentences:
            flags.append("too_few_sentences")
        if word_count > max_words:
            flags.append("too_long")
        if has_repetition:
            flags.append("repetition")

        severe_flags = {"too_short", "too_few_sentences", "incomplete_tail", "repetition"}
        is_valid = not any(flag in severe_flags for flag in flags)
        return OutputQuality(
            word_count=word_count,
            sentence_count=sentence_count,
            ends_cleanly=ends_cleanly,
            has_repetition=has_repetition,
            is_valid=is_valid,
            flags=flags,
        )

    async def _request_once(self, prompt: str) -> tuple[str | None, str | None]:
        return await self._request_once_with_contents(
            [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ]
        )

    async def _request_once_with_contents(self, contents: list[dict]) -> tuple[str | None, str | None]:
        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.settings.gemini_temperature,
                "topP": self.settings.gemini_top_p,
                "maxOutputTokens": self.settings.gemini_max_output_tokens,
            },
        }
        system_prompt = self._system_prompt_text()
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if self.settings.gemini_thinking_budget >= 0:
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": self.settings.gemini_thinking_budget}

        base = self.settings.gemini_endpoint.rstrip("/")
        url = f"{base}/models/{self.settings.gemini_model}:generateContent"
        last_err: str | None = None
        attempts = max(1, self.settings.gemini_retries + 1)
        if not self.configured:
            return None, "Gemini API key is not configured."
        total_attempts = attempts * max(1, self.key_count)
        for attempt in range(total_attempts):
            key, key_id = self._provider_pool.lease(self._provider)
            if not key or not key_id:
                return None, "Gemini API key is not configured."
            headers = {
                "x-goog-api-key": key,
                "content-type": "application/json",
            }
            try:
                async with httpx.AsyncClient(timeout=self.settings.gemini_timeout_seconds) as client:
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
                    continue
                payload = resp.json()
                text = self._extract_text(payload)
                if text:
                    self._provider_pool.report_success(self._provider, key_id)
                    return text, None
                last_err = "Gemini response had no text candidate."
                self._provider_pool.report_failure(self._provider, key_id, status_code=0, error=last_err)
            except Exception as exc:  # pragma: no cover - defensive
                err = str(exc).strip()
                if not err:
                    err = repr(exc)
                last_err = f"Gemini transport error ({exc.__class__.__name__}): {err}"
                self._provider_pool.report_failure(self._provider, key_id, status_code=0, error=last_err)
                if attempt + 1 < total_attempts:
                    await asyncio.sleep(self._retry_delay_seconds(attempt))
        return None, last_err or "Gemini request failed."

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
                    return f"Gemini HTTP {resp.status_code}: {msg.strip()}"
        except Exception:
            pass
        text = resp.text.strip()
        if len(text) > 220:
            text = text[:220].rstrip() + "..."
        return f"Gemini HTTP {resp.status_code}: {text}"

    def _extract_text(self, payload: dict) -> str:
        candidates = payload.get("candidates") or []
        for candidate in candidates:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        return ""

    def _sanitize_text(self, text: str) -> str:
        cleaned = text.replace("\r\n", "\n").strip()
        cleaned = re.sub(r"(?i)^based on (the )?available context,?\s*", "", cleaned).strip()
        cleaned = re.sub(r"(?i)^user (asks|asked|proposed|wants)\s*[:,-]\s*", "", cleaned).strip()
        # Preserve paragraph breaks while normalizing excess whitespace.
        paragraphs = [" ".join(p.split()) for p in re.split(r"\n{2,}", cleaned) if p.strip()]
        if not paragraphs:
            return ""
        cleaned = "\n\n".join(paragraphs)
        if len(_WORD_RE.findall(cleaned)) > 420:
            tokens = re.findall(r"\S+\s*", cleaned)
            out: list[str] = []
            wc = 0
            for token in tokens:
                out.append(token)
                wc += len(_WORD_RE.findall(token))
                if wc >= 420:
                    break
            cleaned = "".join(out).strip()
            if cleaned and cleaned[-1] not in ".!?":
                cleaned = cleaned.rstrip(" ,;:-") + "."
        return cleaned

    def _build_prompt(
        self,
        *,
        user_text: str,
        deterministic_draft: str,
        summary: str,
        recent_messages: Iterable[MessageRecord],
        repair_mode: bool,
        prior_output: str | None,
    ) -> str:
        history_lines = []
        for m in list(recent_messages)[-8:]:
            history_lines.append(f"{m.role}: {m.content}")
        history_text = "\n".join(history_lines)
        system_prompt = self._system_prompt_text()

        if repair_mode:
            return (
                "Continue and fix the previous answer.\n"
                "Keep the same intent, remove obvious repetition, and ensure it ends cleanly.\n\n"
                f"System context:\n{system_prompt or '(none)'}\n\n"
                f"User:\n{user_text}\n\n"
                f"Summary:\n{summary or '(none)'}\n\n"
                f"Recent messages:\n{history_text or '(none)'}\n\n"
                f"Previous answer:\n{prior_output or '(none)'}\n"
            )

        return (
            "You are Fisac, a pragmatic technical partner.\n"
            "Respond in a conversational but concrete way.\n"
            "Avoid generic boilerplate and repeated lead-ins.\n"
            "By default, write a complete answer in 2-4 short paragraphs.\n"
            "If the prompt is technical, include specific recommendation(s) and tradeoffs.\n\n"
            f"System context:\n{system_prompt or '(none)'}\n\n"
            f"User:\n{user_text}\n\n"
            f"Summary:\n{summary or '(none)'}\n\n"
            f"Recent messages:\n{history_text or '(none)'}\n\n"
            f"Reference draft:\n{deterministic_draft}\n"
        )

    def _build_context_prompt(
        self,
        *,
        user_text: str,
        summary: str,
        recent_messages: Iterable[MessageRecord],
        focus: str,
    ) -> str:
        history_lines = []
        for m in list(recent_messages)[-8:]:
            if m.role not in {"user", "assistant"}:
                continue
            history_lines.append(f"{m.role}: {m.content}")
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
        for m in recent_messages[-8:]:
            if m.role not in {"user", "assistant"}:
                continue
            history_lines.append(f"{m.role}: {m.content}")
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
        parts = re.split(r"[\n;]+", raw)
        out: list[str] = []
        seen: set[str] = set()
        user_tokens = self._tokens(user_text)
        for part in parts:
            line = " ".join(part.strip().split())
            line = re.sub(r"^[*\-\d\.\)\(]+\s*", "", line).strip()
            if not line:
                continue
            if self._is_meta_sentence(line):
                continue
            if self._looks_like_echo(line, user_tokens):
                continue
            norm = " ".join(w.lower() for w in _WORD_RE.findall(line))
            if not norm:
                continue
            if len(norm.split()) < 3:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            if line[-1] not in ".!?":
                line += "."
            out.append(line)
            if len(out) >= 10:
                break
        return " ".join(out)

    def _sanitize_candidate_answer(self, text: str, *, user_text: str) -> str:
        cleaned = self._sanitize_text(text)
        if not cleaned:
            return ""
        user_tokens = self._tokens(user_text)
        kept: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", cleaned):
            sentence = " ".join(sentence.strip().split())
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

    def _system_prompt_text(self) -> str:
        if not self.settings.system_prompt_enabled:
            return ""
        return self.settings.system_prompt.strip()

    def _api_keys(self) -> list[str]:
        keys = [k.strip() for k in self.settings.gemini_api_keys if k.strip()]
        if keys:
            return keys
        primary = self.settings.gemini_api_key.strip()
        return [primary] if primary else []
