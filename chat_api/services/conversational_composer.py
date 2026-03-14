from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Sequence

from chat_api.models import MessageRecord

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class ComposeInput:
    user_text: str
    bridge_text: str
    summary: str
    recent_messages: Sequence[MessageRecord]
    tool_text: str | None = None


class ConversationalComposer:
    _INCOMPLETE_TAILS = {
        "and",
        "or",
        "for",
        "with",
        "to",
        "into",
        "onto",
        "then",
        "but",
        "because",
        "if",
    }
    _GENERIC_MEMORY_PHRASES = (
        "i can reason from first principles even before strong memory retrieval is established",
        "share the top constraint",
        "i processed your message and updated my internal state",
        "tell me your goal and constraints",
        "i need more context",
    )
    _NO_BETTER_SIGNAL = "no better signal"

    def classify_intent(self, text: str) -> str:
        raw = text.strip().lower()
        if raw.startswith("/tool"):
            return "tool"
        if any(k in raw for k in ("idea", "brainstorm", "options", "what should", "suggest")):
            return "brainstorm"
        if raw.endswith("?") or any(k in raw for k in ("why", "how", "what", "when", "where")):
            return "question"
        if any(k in raw for k in ("compare", "tradeoff", "pros", "cons")):
            return "analysis"
        return "general"

    def compose(self, payload: ComposeInput) -> str:
        if payload.tool_text:
            return self._sanitize(payload.tool_text)

        user_text = payload.user_text.strip()
        bridge = self._sanitize(payload.bridge_text) if payload.bridge_text.strip() else ""
        if bridge and not self._is_generic_memory_text(bridge) and self._has_topic_overlap(user_text, bridge):
            return bridge

        summary = self._sanitize(payload.summary) if payload.summary.strip() else ""
        if summary and self._has_topic_overlap(user_text, summary):
            return summary

        recent_user = ""
        for msg in reversed(payload.recent_messages):
            if msg.role == "user" and msg.content.strip():
                recent_user = msg.content.strip()
                break

        if user_text and recent_user and recent_user.lower() != user_text.lower() and len(_WORD_RE.findall(user_text)) <= 3:
            return self._sanitize(f"{user_text}. {recent_user}.")
        return self._NO_BETTER_SIGNAL

    def sanitize_generated(self, text: str) -> str:
        return self._sanitize(text)

    def choose_generated_or_fallback(self, generated: str, deterministic: str) -> str:
        candidate = self._sanitize(generated)
        fallback = self._sanitize(deterministic)
        if not candidate:
            return fallback
        if len(_WORD_RE.findall(candidate)) < 24:
            return fallback
        if self._jaccard(self._norm(candidate), self._norm(fallback)) > 0.92:
            return fallback
        return candidate

    def is_incomplete_tail(self, text: str) -> bool:
        raw = text.rstrip()
        if not raw:
            return True
        if raw.endswith("-") or raw.endswith(":"):
            return True
        if raw[-1] not in ".!?":
            last_word = self._last_word(raw).lower()
            if last_word in self._INCOMPLETE_TAILS:
                return True
            if len(_WORD_RE.findall(raw)) >= 8:
                return True
        return False

    def _sanitize(self, text: str) -> str:
        text = text.replace("\r\n", "\n").strip()
        if not text:
            return ""

        paragraphs = [self._collapse_spaces(p) for p in re.split(r"\n{2,}", text) if p.strip()]
        seen_norms: list[str] = []
        deduped_paragraphs: list[str] = []
        for paragraph in paragraphs:
            kept_sentences: list[str] = []
            for sentence in [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]:
                norm = self._norm(sentence)
                if not norm:
                    continue
                duplicate = False
                for seen in seen_norms:
                    if self._sentence_similarity(norm, seen) >= 0.96:
                        duplicate = True
                        break
                if duplicate:
                    continue
                seen_norms.append(norm)
                kept_sentences.append(sentence)
            if kept_sentences:
                deduped_paragraphs.append(" ".join(kept_sentences))

        merged = "\n\n".join(deduped_paragraphs) if deduped_paragraphs else self._collapse_spaces(text)
        merged = self._strip_leading_labels(merged)

        # Collapse accidental duplicated lead-ins without cutting full tails.
        for prefix in ("here is what i infer:", "my read:", "here is my take:", "quick read:", "what stands out:"):
            pattern = re.compile(rf"(?:{re.escape(prefix)}\s*){{2,}}", flags=re.IGNORECASE)
            merged = pattern.sub(prefix.title() + " ", merged)
        merged = re.sub(
            r"(?:i processed your message and updated my internal state\.?\s*){2,}",
            "I processed your message and updated my internal state. ",
            merged,
            flags=re.IGNORECASE,
        )

        words = _WORD_RE.findall(merged)
        if len(words) > 320:
            # Keep balanced-size outputs while preserving clean ending.
            cutoff = 320
            tokens = re.findall(r"\S+\s*", merged)
            out: list[str] = []
            wc = 0
            for token in tokens:
                out.append(token)
                wc += len(_WORD_RE.findall(token))
                if wc >= cutoff:
                    break
            merged = "".join(out).strip()
            if self.is_incomplete_tail(merged):
                merged = merged.rstrip(" ,;:-") + "."
        elif self.is_incomplete_tail(merged):
            merged = merged.rstrip(" ,;:-") + "."
        return merged

    def _collapse_spaces(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _norm(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def _jaccard(self, a: str, b: str) -> float:
        sa = {tok for tok in a.split(" ") if tok}
        sb = {tok for tok in b.split(" ") if tok}
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / max(1, union)

    def _sentence_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    def _last_word(self, text: str) -> str:
        words = _WORD_RE.findall(text)
        return words[-1] if words else ""

    def _strip_leading_labels(self, text: str) -> str:
        out = text.strip()
        while True:
            next_out = re.sub(
                r"^\s*(?:answer|why|next\s*steps?|here\s+is\s+what\s+i\s+infer)\s*:\s*",
                "",
                out,
                flags=re.IGNORECASE,
            ).strip()
            if next_out == out:
                break
            out = next_out
        return out

    def _is_generic_memory_text(self, text: str) -> bool:
        low = self._norm(text)
        if not low:
            return True
        for phrase in self._GENERIC_MEMORY_PHRASES:
            if phrase in low:
                return True
        return False

    def _has_topic_overlap(self, a: str, b: str) -> bool:
        na = self._norm(a)
        nb = self._norm(b)
        if not na or not nb:
            return False
        return self._jaccard(na, nb) >= 0.08
