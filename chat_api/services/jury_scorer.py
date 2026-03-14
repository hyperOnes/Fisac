from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from chat_api.models import MessageRecord
from chat_api.services.reasoning_orchestrator import ReasoningCandidate

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "if", "in", "is", "it", "of",
    "on", "or", "that", "the", "to", "we", "what", "when", "where", "which", "why", "with", "you", "your",
}
_ACTION_WORDS = {
    "build", "test", "measure", "compare", "prototype", "implement", "validate", "rank", "choose", "prioritize",
    "estimate", "simulate", "design", "verify", "deploy",
}


@dataclass
class JuryScore:
    index: int
    total: float
    coverage: float
    echo: float
    actionability: float
    coherence: float
    alignment: float
    evidence: float
    contradiction_penalty: float
    meta_penalty: float
    generic_penalty: float


class JuryScorer:
    def __init__(self, *, learned_head_path: str = "", learned_heads_enabled: bool = False) -> None:
        self.learned_heads_enabled = bool(learned_heads_enabled)
        self._feature_names = [
            "coverage",
            "echo",
            "actionability",
            "coherence",
            "alignment",
            "evidence",
            "contradiction_penalty",
            "meta_penalty",
            "generic_penalty",
            "role_bonus",
            "role_mismatch_penalty",
            "length_quality",
            "score_hint",
        ]
        self._head_w: torch.Tensor | None = None
        self._head_b: float = 0.0
        self._head_version: str | None = None
        if self.learned_heads_enabled and learned_head_path.strip():
            self._load_learned_head(learned_head_path.strip())

    @property
    def head_version(self) -> str | None:
        return self._head_version

    def choose_best(
        self,
        *,
        user_text: str,
        candidates: Sequence[ReasoningCandidate],
        recent_messages: Sequence[MessageRecord],
        query_vec: torch.Tensor,
        encode_text: Callable[[str], torch.Tensor],
    ) -> tuple[JuryScore | None, list[JuryScore]]:
        if not candidates:
            return None, []

        user_tokens = self._tokens(user_text)
        last_assistant = ""
        for msg in reversed(recent_messages):
            if msg.role == "assistant" and msg.content.strip():
                last_assistant = msg.content
                break

        scores: list[JuryScore] = []
        for idx, candidate in enumerate(candidates):
            text = candidate.text.strip()
            if not text:
                continue

            cand_tokens = self._tokens(text)
            coverage = self._token_overlap(cand_tokens, user_tokens)
            echo = self._echo_score(text, user_text)
            actionability = self._actionability(cand_tokens)
            coherence = self._coherence(text, last_assistant)
            alignment = self._alignment(text, query_vec, encode_text)
            if coverage < 0.08:
                alignment *= 0.55
            meta_penalty = self._meta_penalty(text)
            length_quality = self._length_quality(cand_tokens)
            generic_penalty = self._generic_penalty(text=text, tokens=cand_tokens, coverage=coverage, actionability=actionability)
            role_bonus = self._role_bonus(candidate.role, user_text)
            role_mismatch_penalty = self._role_mismatch_penalty(candidate.role, user_text)
            evidence = self._evidence_score(cand_tokens)
            contradiction_penalty = self._contradiction_penalty(text)

            learned_boost = self._learned_head_score(
                {
                    "coverage": coverage,
                    "echo": echo,
                    "actionability": actionability,
                    "coherence": coherence,
                    "alignment": alignment,
                    "evidence": evidence,
                    "contradiction_penalty": contradiction_penalty,
                    "meta_penalty": meta_penalty,
                    "generic_penalty": generic_penalty,
                    "role_bonus": role_bonus,
                    "role_mismatch_penalty": role_mismatch_penalty,
                    "length_quality": length_quality,
                    "score_hint": float(candidate.score_hint),
                }
            )

            total = (
                0.34 * coverage
                + 0.18 * actionability
                + 0.12 * coherence
                + 0.20 * alignment
                + 0.10 * length_quality
                + 0.08 * evidence
                - 0.30 * contradiction_penalty
                - 0.45 * echo
                - 0.42 * meta_penalty
                - 0.34 * generic_penalty
                + 0.12 * candidate.score_hint
                + 0.24 * role_bonus
                - 0.24 * role_mismatch_penalty
                + learned_boost
            )
            if echo > 0.92:
                total -= 0.8
            if contradiction_penalty > 0.7:
                total -= 0.5
            if meta_penalty > 0.7:
                total -= 0.5
            if generic_penalty > 0.7:
                total -= 0.4
            scores.append(
                JuryScore(
                    index=idx,
                    total=float(total),
                    coverage=float(coverage),
                    echo=float(echo),
                    actionability=float(actionability),
                    coherence=float(coherence),
                    alignment=float(alignment),
                    evidence=float(evidence),
                    contradiction_penalty=float(contradiction_penalty),
                    meta_penalty=float(meta_penalty),
                    generic_penalty=float(generic_penalty),
                )
            )

        if not scores:
            return None, []

        scores.sort(key=lambda s: s.total, reverse=True)
        return scores[0], scores

    def _tokens(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def _token_overlap(self, a: Sequence[str], b: Sequence[str]) -> float:
        sa = {t for t in a if t and t not in _STOP}
        sb = {t for t in b if t and t not in _STOP}
        if not sa:
            sa = {t for t in a if t}
        if not sb:
            sb = {t for t in b if t}
        if not sa or not sb:
            return 0.0
        return float(len(sa & sb) / max(1, len(sb)))

    def _echo_score(self, candidate: str, user_text: str) -> float:
        ct = set(self._tokens(candidate))
        ut = set(self._tokens(user_text))
        if not ct or not ut:
            return 0.0
        j = len(ct & ut) / max(1, len(ct | ut))
        starts = " ".join(self._tokens(candidate)[:8]) == " ".join(self._tokens(user_text)[:8])
        return float(j + (0.3 if starts else 0.0))

    def _actionability(self, tokens: Sequence[str]) -> float:
        if not tokens:
            return 0.0
        hits = sum(1 for t in tokens if t in _ACTION_WORDS)
        return min(1.0, hits / 3.0)

    def _length_quality(self, tokens: Sequence[str]) -> float:
        n = len(tokens)
        if n < 10:
            return 0.0
        if 18 <= n <= 90:
            return 1.0
        if n < 18:
            return max(0.0, n / 18.0)
        return max(0.0, 1.0 - ((n - 90) / 180.0))

    def _meta_penalty(self, text: str) -> float:
        low = " ".join(text.lower().split())
        patterns = (
            "based on the available context",
            "user asks",
            "user query",
            "the user asks",
            "user wants",
            "user proposed",
            "user demands",
            "user requests",
            "active topic terms",
            "core concepts involve",
            "no better signal",
            "strongest signals right now",
        )
        hits = sum(1 for p in patterns if p in low)
        return min(1.0, hits / 2.0)

    def _generic_penalty(self, *, text: str, tokens: Sequence[str], coverage: float, actionability: float) -> float:
        low = " ".join(text.lower().split())
        penalty = 0.0
        if len(tokens) < 14:
            penalty += 0.35
        if coverage < 0.05:
            penalty += 0.35
        if coverage < 0.08 and actionability < 0.34:
            penalty += 0.25
        generic_patterns = (
            "choose the path with fewer moving parts",
            "need more context",
            "provide more detail",
            "best option depends",
            "strongest signals right now",
            "primary terms:",
            "infer objective from recent turns",
        )
        if low.startswith("need ") and "pass/fail" not in low and "threshold" not in low:
            penalty += 0.35
        if low.startswith("user "):
            penalty += 0.65
        if any(p in low for p in generic_patterns):
            penalty += 1.0
        return min(1.0, penalty)

    def _contradiction_penalty(self, text: str) -> float:
        low = text.lower()
        tokens = set(self._tokens(low))
        pairs = (
            ("always", "never"),
            ("must", "optional"),
            ("easier", "harder"),
            ("feasible", "impossible"),
            ("safe", "unsafe"),
            ("reliable", "unreliable"),
        )
        hits = 0
        for a, b in pairs:
            if a in tokens and b in tokens:
                hits += 1
        if "on the other hand" in low and "therefore" in low:
            hits += 1
        return min(1.0, hits / 2.0)

    def _role_bonus(self, role: str, user_text: str) -> float:
        role = role.strip().lower()
        low = user_text.lower()
        bonus = 0.0
        if any(k in low for k in ("how", "exactly", "steps", "plan", "solve")):
            if role == "implementation_sequence":
                bonus += 1.0
            if role == "failure_modes":
                bonus += 0.5
            if role == "direct_answer":
                bonus += 0.25
        if any(k in low for k in ("continue", "next", "then")):
            if role == "implementation_sequence":
                bonus += 0.85
            if role == "failure_modes":
                bonus += 0.35
        if any(k in low for k in ("why", "explain")):
            if role in {"failure_modes", "skeptical_counterexample"}:
                bonus += 0.8
            if role == "direct_answer":
                bonus += 0.3
        if any(k in low for k in (" or ", " vs ", " versus ", "which", "easiest", "compare")):
            if role == "comparative_feasibility":
                bonus += 1.0
            if role == "direct_answer":
                bonus += 0.5
        return max(0.0, min(1.0, bonus))

    def _role_mismatch_penalty(self, role: str, user_text: str) -> float:
        role = role.strip().lower()
        low = user_text.lower()
        penalty = 0.0
        if any(k in low for k in ("how", "exactly", "steps", "plan", "solve")):
            if role not in {"implementation_sequence", "direct_answer"}:
                penalty += 0.8
        if any(k in low for k in ("continue", "next", "then")):
            if role not in {"implementation_sequence", "failure_modes", "direct_answer"}:
                penalty += 0.7
        if any(k in low for k in ("why", "explain")):
            if role == "implementation_sequence":
                penalty += 0.4
        if any(k in low for k in (" or ", " vs ", " versus ", "which", "easiest", "compare")):
            if role == "failure_modes":
                penalty += 0.3
        return max(0.0, min(1.0, penalty))

    def _evidence_score(self, tokens: Sequence[str]) -> float:
        if not tokens:
            return 0.0
        evidence_words = {
            "because",
            "therefore",
            "evidence",
            "constraint",
            "tradeoff",
            "risk",
            "failure",
            "mitigation",
            "test",
            "measure",
            "threshold",
            "abort",
            "phase",
            "step",
        }
        hits = sum(1 for t in tokens if t in evidence_words)
        return min(1.0, hits / 4.0)

    def _coherence(self, candidate: str, last_assistant: str) -> float:
        if not last_assistant.strip():
            return 0.6
        ct = self._tokens(candidate)
        lt = self._tokens(last_assistant)
        overlap = self._token_overlap(ct, lt)
        # Encourage some continuity but penalize full repetition.
        if overlap > 0.92:
            return 0.0
        if overlap < 0.05:
            return 0.35
        return float(0.75 - abs(0.35 - overlap))

    def _alignment(self, text: str, query_vec: torch.Tensor, encode_text: Callable[[str], torch.Tensor]) -> float:
        cand = encode_text(text)[0].detach().float().cpu().reshape(-1)
        q = query_vec.detach().float().cpu().reshape(-1)
        if cand.numel() != q.numel():
            return 0.0
        denom = cand.norm().item() * q.norm().item()
        if denom <= 1e-8:
            return 0.0
        sim = float(F.cosine_similarity(cand.unsqueeze(0), q.unsqueeze(0)).item())
        return max(0.0, min(1.0, (sim + 1.0) * 0.5))

    def _load_learned_head(self, head_path: str) -> None:
        path = Path(head_path)
        if not path.exists():
            return
        try:
            payload = torch.load(path, map_location="cpu")
            weights = payload.get("weights")
            bias = float(payload.get("bias", 0.0))
            version = str(payload.get("version", "learned"))
            feature_names = payload.get("feature_names")
            if isinstance(feature_names, list) and feature_names:
                self._feature_names = [str(x) for x in feature_names]
            if isinstance(weights, list) and weights:
                self._head_w = torch.tensor([float(x) for x in weights], dtype=torch.float32)
                self._head_b = bias
                self._head_version = version
                return
            if isinstance(weights, torch.Tensor) and weights.numel() > 0:
                self._head_w = weights.detach().float().cpu().reshape(-1)
                self._head_b = bias
                self._head_version = version
                return
            sidecar = path.with_suffix(".json")
            if sidecar.exists():
                alt = json.loads(sidecar.read_text(encoding="utf-8"))
                w = alt.get("weights")
                if isinstance(w, list) and w:
                    self._head_w = torch.tensor([float(x) for x in w], dtype=torch.float32)
                    self._head_b = float(alt.get("bias", 0.0))
                    self._head_version = str(alt.get("version", "learned_json"))
        except Exception:
            self._head_w = None
            self._head_version = None

    def _learned_head_score(self, features: dict[str, float]) -> float:
        if not self.learned_heads_enabled or self._head_w is None:
            return 0.0
        vec = torch.tensor([float(features.get(name, 0.0)) for name in self._feature_names], dtype=torch.float32)
        if vec.numel() != self._head_w.numel():
            return 0.0
        raw = float(torch.dot(vec, self._head_w).item() + self._head_b)
        # Bound the learned contribution to keep heuristic dominance and avoid instability.
        return max(-0.35, min(0.35, raw))
