from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Sequence

from chat_api.models import MessageRecord

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_RE = re.compile(r"[a-z0-9']+")


class SummaryService:
    def __init__(self, max_sentences: int = 3, max_chars: int = 700) -> None:
        self.max_sentences = max(1, max_sentences)
        self.max_chars = max_chars

    def update_summary(self, messages: Sequence[MessageRecord], previous_summary: str = "") -> str:
        text = "\n".join(f"{m.role}: {m.content}" for m in messages if m.content.strip())
        if previous_summary.strip():
            text = previous_summary.strip() + "\n" + text

        sentences = self._split_sentences(text)
        if not sentences:
            return previous_summary.strip()

        tokenized = [self._tokens(s) for s in sentences]
        doc_freq: defaultdict[str, int] = defaultdict(int)
        for tokens in tokenized:
            for tok in set(tokens):
                doc_freq[tok] += 1

        n_docs = float(len(sentences))
        scored: list[tuple[float, int, str]] = []
        for i, (sentence, tokens) in enumerate(zip(sentences, tokenized)):
            if not tokens:
                continue
            tf = Counter(tokens)
            score = 0.0
            for tok, cnt in tf.items():
                idf = math.log((1.0 + n_docs) / (1.0 + float(doc_freq[tok]))) + 1.0
                score += float(cnt) * idf
            # Slightly favor newer context while keeping deterministic behavior.
            recency = 1.0 + (i / max(1.0, n_docs - 1.0)) * 0.1
            score *= recency
            scored.append((score, i, sentence.strip()))

        if not scored:
            return previous_summary.strip()

        scored.sort(key=lambda item: item[0], reverse=True)
        top = sorted(scored[: self.max_sentences], key=lambda item: item[1])
        summary = " ".join(item[2] for item in top).strip()
        if len(summary) > self.max_chars:
            summary = summary[: self.max_chars].rstrip() + "..."
        return summary

    def _split_sentences(self, text: str) -> list[str]:
        parts = [p.strip() for p in _SENTENCE_RE.split(text) if p and p.strip()]
        return parts

    def _tokens(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())
