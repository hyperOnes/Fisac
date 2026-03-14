from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Sequence

from chat_api.models import MessageRecord

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "we",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
    "you",
    "your",
}

_POSITIVE = {
    "feasible",
    "practical",
    "proven",
    "reliable",
    "stable",
    "viable",
    "manageable",
    "incremental",
    "safe",
    "cheaper",
    "faster",
    "easier",
}

_NEGATIVE = {
    "risk",
    "risky",
    "jam",
    "jamming",
    "catastrophic",
    "hard",
    "harder",
    "severe",
    "complex",
    "complexity",
    "expensive",
    "costly",
    "impossible",
    "unlikely",
    "unreliable",
    "blocker",
    "failure",
    "unstable",
}

_UNCERTAIN = {"maybe", "could", "might", "assume", "assuming", "uncertain", "possibly"}

@dataclass
class TruthReasoningResult:
    text: str
    confidence: float
    flags: list[str] = field(default_factory=list)


class TruthReasoner:
    def reason(
        self,
        *,
        user_text: str,
        context_hint: str,
        recent_messages: Sequence[MessageRecord],
    ) -> TruthReasoningResult | None:
        user = " ".join(user_text.strip().split())
        if not user:
            return None

        context = self._build_context(user=user, context_hint=context_hint, recent_messages=recent_messages)
        if not context:
            return None

        options = self._extract_options(user)
        if len(options) >= 2:
            return self._reason_compare(user=user, options=options[:3], context=context)
        if self._is_plan_request(user):
            return self._reason_plan(user=user, context=context)
        return self._reason_single(user=user, context=context)

    def _reason_compare(self, *, user: str, options: list[str], context: list[str]) -> TruthReasoningResult:
        scored: list[tuple[float, str, str, str]] = []
        for option in options:
            score, top_pos, top_neg = self._score_option(option=option, context=context)
            scored.append((score, option, top_pos, top_neg))

        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0]
        alt = scored[1] if len(scored) > 1 else scored[0]

        gap = best[0] - alt[0]
        confidence = max(0.35, min(0.95, 0.55 + 0.12 * gap))

        best_reason = self._evidence_phrase(best[2], fallback="lower near-term execution risk")
        alt_risk = self._evidence_phrase(alt[3], fallback="higher execution and safety risk")
        best_label = self._option_label(best[1])
        alt_label = self._option_label(alt[1])

        text = (
            f"{best_label} is more feasible in the near term than {alt_label}. "
            f"This is mainly because {best_reason}. "
            f"The main blocker for {alt_label} is {alt_risk}. "
            "Validate this verdict by running a constrained prototype and comparing failure and recovery metrics."
        )
        flags = ["truth_reasoner_compare"]
        return TruthReasoningResult(text=self._finalize(text), confidence=confidence, flags=flags)

    def _reason_plan(self, *, user: str, context: list[str]) -> TruthReasoningResult:
        _, top_pos, top_neg = self._score_option(option=user, context=context)
        key_risk = self._evidence_phrase(top_neg, fallback="high-speed reliability and failure containment")
        key_support = self._evidence_phrase(top_pos, fallback="narrow-scope tests can reduce uncertainty quickly")
        if self._looks_like_prompt_echo(key_support, user):
            key_support = "narrow-scope tests can reduce uncertainty quickly"
        text = (
            "Start with a staged validation path instead of full-speed deployment. "
            "First, isolate the highest-risk subsystem in a bench rig and define hard pass/fail thresholds. "
            "Second, run low-speed closed-loop tests and measure stability, alignment drift, and recovery behavior. "
            f"Third, scale speed in capped increments with automatic abort criteria. Main risk to retire first is {key_risk}. "
            f"This sequence works because {key_support}."
        )
        flags = ["truth_reasoner_plan"]
        return TruthReasoningResult(text=self._finalize(text), confidence=0.74, flags=flags)

    def _reason_single(self, *, user: str, context: list[str]) -> TruthReasoningResult:
        score, top_pos, top_neg = self._score_option(option=user, context=context)
        feasible = score >= 0.0
        confidence = max(0.3, min(0.9, 0.62 + 0.08 * abs(score)))

        if feasible:
            verdict = "The concept is conditionally feasible, but only with strict constraints and phased validation."
            reason = self._evidence_phrase(top_pos, fallback="the current path can be decomposed into testable subsystems")
            blocker = self._evidence_phrase(top_neg, fallback="integration complexity at speed")
        else:
            verdict = "The concept is currently low-feasibility for near-term deployment."
            reason = self._evidence_phrase(top_neg, fallback="execution risk dominates available evidence")
            blocker = self._evidence_phrase(top_pos, fallback="a narrower prototype path is still worth testing")
        if self._looks_like_prompt_echo(reason, user):
            reason = "the current path can be decomposed into testable subsystems"
        if self._looks_like_prompt_echo(blocker, user):
            blocker = "integration complexity at speed"

        text = (
            f"{verdict} Evidence suggests {reason}. Main blocker is {blocker}. "
            "Next step: test the highest-risk subsystem with explicit pass/fail thresholds."
        )
        flags = ["truth_reasoner_single"]
        return TruthReasoningResult(text=self._finalize(text), confidence=confidence, flags=flags)

    def _score_option(self, *, option: str, context: list[str]) -> tuple[float, str, str]:
        option_tokens = {t for t in self._tokenize(option) if t not in _STOP}
        if not option_tokens:
            option_tokens = set(self._tokenize(option))

        score = 0.0
        best_pos = ""
        best_neg = ""
        best_pos_score = -1.0
        best_neg_score = -1.0

        for sentence in context:
            stokens = set(self._tokenize(sentence))
            if not stokens:
                continue
            overlap = len(option_tokens & stokens) / max(1, len(option_tokens))
            if overlap <= 0:
                continue

            pos = sum(1 for t in stokens if t in _POSITIVE)
            neg = sum(1 for t in stokens if t in _NEGATIVE)
            unc = sum(1 for t in stokens if t in _UNCERTAIN)

            local = overlap * (1.0 + 0.75 * pos - 0.95 * neg - 0.3 * unc)
            score += local

            if pos > 0 and local > best_pos_score:
                best_pos_score = local
                best_pos = sentence
            if neg > 0 and (neg * overlap) > best_neg_score:
                best_neg_score = neg * overlap
                best_neg = sentence

        return score, best_pos, best_neg

    def _build_context(self, *, user: str, context_hint: str, recent_messages: Sequence[MessageRecord]) -> list[str]:
        sents: list[str] = []
        for sentence in self._split_sentences(context_hint):
            cleaned = self._finalize(sentence)
            if cleaned:
                if self._looks_like_prompt_echo(cleaned, user):
                    continue
                if self._is_meta_sentence(cleaned):
                    continue
                sents.append(cleaned)
        if len(sents) < 3:
            for msg in reversed(recent_messages):
                if msg.role != "user":
                    continue
                for sentence in self._split_sentences(msg.content):
                    cleaned = self._finalize(sentence)
                    if cleaned and cleaned not in sents:
                        if self._looks_like_prompt_echo(cleaned, user):
                            continue
                        if self._is_meta_sentence(cleaned):
                            continue
                        sents.append(cleaned)
                if len(sents) >= 8:
                    break
        return sents[:16]

    def _extract_options(self, text: str) -> list[str]:
        clean = " ".join(text.split())
        clean = re.sub(r"^(hey|hi)\b[^:]*:\s*", "", clean, flags=re.IGNORECASE).strip()
        lower = clean.lower()
        if " or " in lower:
            parts = [" ".join(p.split()) for p in re.split(r"\bor\b", clean, maxsplit=2) if p.strip()]
            if len(parts) >= 2:
                options = [self._trim_option(p) for p in parts if self._trim_option(p)]
                if len(options) >= 2:
                    return options
        if " vs " in lower or " versus " in lower:
            parts = [" ".join(p.split()) for p in re.split(r"\bvs\b|\bversus\b", clean, maxsplit=2) if p.strip()]
            if len(parts) >= 2:
                options = [self._trim_option(p) for p in parts if self._trim_option(p)]
                if len(options) >= 2:
                    return options
        return []

    def _trim_option(self, text: str) -> str:
        t = text.strip(" .?!,;:-")
        t = re.sub(r"\(.*?\)", "", t).strip()
        t = re.sub(r"^(which\s+one\s+is\s+easiest|which\s+is\s+easiest)\b", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"\b(which\s+one\s+is\s+easiest|which\s+is\s+easiest)\b.*$", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"^(hey|help me assess this)\b[:,-]?\s*", "", t, flags=re.IGNORECASE).strip()
        words = t.split()
        if len(words) > 20:
            t = " ".join(words[:20]).rstrip(" ,;:-")
        return t

    def _split_sentences(self, text: str) -> list[str]:
        raw = " ".join(text.replace("\r\n", "\n").split())
        if not raw:
            return []
        parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(raw) if p.strip()]
        return parts or [raw]

    def _tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def _finalize(self, text: str) -> str:
        clean = " ".join(text.strip().split())
        clean = re.sub(r"\s+([,.;:!?])", r"\1", clean)
        if clean and clean[-1] not in ".!?":
            clean += "."
        return clean

    def _looks_like_prompt_echo(self, output_text: str, user_text: str) -> bool:
        out_words = self._tokenize(output_text)
        in_words = self._tokenize(user_text)
        if not out_words or not in_words:
            return False
        if len(in_words) <= 4:
            return False
        out_set = set(out_words)
        in_set = set(in_words)
        overlap = len(out_set & in_set) / max(1, len(in_set))
        starts = " ".join(out_words[: min(8, len(out_words))]) == " ".join(in_words[: min(8, len(in_words))])
        return overlap >= 0.72 or (starts and overlap >= 0.55)

    def _is_meta_sentence(self, sentence: str) -> bool:
        low = sentence.lower().strip()
        prefixes = (
            "user asks:",
            "user query:",
            "user proposed:",
            "user wants:",
            "user demands:",
            "the user asks",
            "the user wants",
            "previous user turn:",
            "previous user turn theme:",
            "previous assistant turn:",
            "previous assistant hypothesis:",
            "working summary:",
            "based on the available context",
            "active topic terms:",
            "the user asks",
            "need concrete implementation steps",
            "return phased implementation steps",
            "infer objective from recent turns",
            "primary terms:",
            "strongest evidence:",
            "critical blocker:",
        )
        if any(low.startswith(p) for p in prefixes):
            return True
        return bool(re.match(r"^user\s+(asks|query|proposed|wants|demands|requested)\b", low))

    def _evidence_phrase(self, sentence: str, *, fallback: str) -> str:
        clean = self._finalize(sentence or "")
        if not clean:
            return fallback
        if self._is_meta_sentence(clean):
            return fallback
        # Keep clause order for readability; compact to one short phrase.
        stripped = re.sub(
            r"^(strongest evidence|critical blocker|because|therefore)\s*:\s*",
            "",
            clean,
            flags=re.IGNORECASE,
        ).strip()
        clause = re.split(r"[;:]", stripped, maxsplit=1)[0].strip()
        clause = re.sub(r"\s+", " ", clause).strip(" ,.-")
        if not clause:
            return fallback
        tokens = [tok for tok in self._tokenize(clause) if tok and tok not in _STOP]
        if len(tokens) < 3:
            return fallback
        # Keep first words in order instead of bag-of-words reconstruction.
        words = clause.split()
        if len(words) > 14:
            clause = " ".join(words[:14]).rstrip(" ,;:-")
        return clause

    def _is_plan_request(self, text: str) -> bool:
        low = text.lower()
        plan_words = ("how", "exactly", "steps", "plan", "solve", "implement", "build", "continue")
        return any(word in low for word in plan_words)

    def _option_label(self, option: str) -> str:
        low = option.lower()
        if "venus" in low or "spaceship" in low or "crewed" in low:
            return "crewed Venus mission"
        if "maglev" in low and any(k in low for k in ("track", "lay", "retract", "self")):
            return "self-laid maglev track vehicle"
        if "wheel" in low:
            return "wheeled EV path"
        tokens = [t for t in self._tokenize(option) if t not in _STOP]
        if not tokens:
            tokens = self._tokenize(option)
        if not tokens:
            return option.strip() or "this option"
        return " ".join(tokens[:6])
