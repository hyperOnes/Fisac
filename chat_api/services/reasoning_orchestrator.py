from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import re
from typing import Sequence

from chat_api.models import MessageRecord
from chat_api.services.gemini_client import GeminiClient
from chat_api.services.openai_client import OpenAIClient


@dataclass
class ReasoningCandidate:
    role: str
    text: str
    provider: str
    score_hint: float = 0.0
    flags: list[str] = field(default_factory=list)


class ReasoningOrchestrator:
    """Generate multiple reasoning candidates from available worker models."""

    ROLES: tuple[str, ...] = (
        "direct_answer",
        "comparative_feasibility",
        "failure_modes",
        "implementation_sequence",
        "skeptical_counterexample",
    )

    def __init__(self, gemini_client: GeminiClient | None, openai_client: OpenAIClient | None) -> None:
        self.gemini_client = gemini_client
        self.openai_client = openai_client

    async def generate_candidates(
        self,
        *,
        user_text: str,
        summary: str,
        recent_messages: Sequence[MessageRecord],
        external_context: str,
        timeout_seconds: float = 8.0,
    ) -> tuple[list[ReasoningCandidate], list[str]]:
        candidates: list[ReasoningCandidate] = []
        flags: list[str] = []

        providers: list[str] = []
        if self.gemini_client is not None and self.gemini_client.configured:
            providers.append("gemini")
        if self.openai_client is not None and self.openai_client.configured:
            providers.append("openai")

        if not providers:
            return [], ["orchestrator_no_provider"]

        tasks: list[asyncio.Task] = []
        task_meta: list[tuple[str, str]] = []
        for role in self.ROLES:
            for provider in providers:
                if provider == "gemini" and self.gemini_client is not None and hasattr(self.gemini_client, "answer_candidate"):
                    task_meta.append((provider, role))
                    tasks.append(
                        asyncio.create_task(
                            self.gemini_client.answer_candidate(
                                user_text=user_text,
                                summary=summary,
                                recent_messages=recent_messages,
                                role=role,
                                external_context=external_context,
                            )
                        )
                    )
                elif provider == "openai" and self.openai_client is not None and hasattr(self.openai_client, "answer_candidate"):
                    task_meta.append((provider, role))
                    tasks.append(
                        asyncio.create_task(
                            self.openai_client.answer_candidate(
                                user_text=user_text,
                                summary=summary,
                                recent_messages=recent_messages,
                                role=role,
                                external_context=external_context,
                            )
                        )
                    )

        if not tasks:
            return [], ["orchestrator_no_tasks"]

        done = await asyncio.gather(*tasks, return_exceptions=True)
        seen_text: set[str] = set()
        role_candidates: dict[str, list[ReasoningCandidate]] = {role: [] for role in self.ROLES}
        for (provider, role), result in zip(task_meta, done):
            if isinstance(result, Exception):
                flags.append(f"orchestrator_{provider}_{role}_error")
                continue
            if getattr(result, "ok", False) and getattr(result, "answer", ""):
                answer = getattr(result, "answer")
                compliance = self._role_compliance(role, answer)
                if compliance < 0.45:
                    flags.append(f"orchestrator_{provider}_{role}_low_compliance")
                    continue
                quality = self._candidate_quality(answer)
                role_candidates[role].append(
                    ReasoningCandidate(
                        role=role,
                        text=answer,
                        provider=provider,
                        score_hint=0.30 + (0.70 * ((0.72 * compliance) + (0.28 * quality))),
                    )
                )
            else:
                flags.append(f"orchestrator_{provider}_{role}_empty")

        # Select strongest candidate per role after provider competition.
        for role in self.ROLES:
            if not role_candidates[role]:
                continue
            role_candidates[role].sort(key=lambda c: c.score_hint, reverse=True)
            picked = role_candidates[role][0]
            norm = self._norm_text(picked.text)
            if norm and norm in seen_text:
                flags.append(f"orchestrator_{picked.provider}_{role}_duplicate")
                continue
            if norm:
                seen_text.add(norm)
            candidates.append(picked)

        # Timeout fallback: if candidates unexpectedly empty, try one quick direct pass.
        if not candidates:
            quick_provider = providers[0]
            try:
                if quick_provider == "gemini" and self.gemini_client is not None:
                    res = await asyncio.wait_for(
                        self.gemini_client.answer_candidate(
                            user_text=user_text,
                            summary=summary,
                            recent_messages=recent_messages,
                            role="direct_answer",
                            external_context=external_context,
                        ),
                        timeout=timeout_seconds,
                    )
                elif quick_provider == "openai" and self.openai_client is not None:
                    res = await asyncio.wait_for(
                        self.openai_client.answer_candidate(
                            user_text=user_text,
                            summary=summary,
                            recent_messages=recent_messages,
                            role="direct_answer",
                            external_context=external_context,
                        ),
                        timeout=timeout_seconds,
                    )
                else:
                    res = None
                if res is not None and res.ok and res.answer:
                    compliance = self._role_compliance("direct_answer", res.answer)
                    if compliance < 0.45:
                        flags.append("orchestrator_quick_retry_low_compliance")
                        return candidates, sorted(set(flags))
                    candidates.append(
                        ReasoningCandidate(
                            role="direct_answer",
                            text=res.answer,
                            provider=quick_provider,
                            score_hint=0.35 + 0.65 * compliance,
                        )
                    )
                else:
                    flags.append("orchestrator_quick_retry_empty")
            except Exception:
                flags.append("orchestrator_quick_retry_error")

        return candidates, sorted(set(flags))

    def _norm_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9']+", " ", text.lower()).strip()

    def _role_compliance(self, role: str, text: str) -> float:
        low = text.lower()
        stripped = " ".join(low.split())
        if (
            stripped.startswith("user ")
            or stripped.startswith("based on the available context")
            or "strongest signals right now" in stripped
            or stripped.startswith("primary terms:")
        ):
            return 0.0
        score = 0.4
        if role == "implementation_sequence":
            if any(k in low for k in ("phase", "step", "first", "then", "next")):
                score += 0.5
        elif role == "failure_modes":
            if any(k in low for k in ("risk", "failure", "jam", "mitigate", "hazard")):
                score += 0.45
        elif role == "comparative_feasibility":
            if any(k in low for k in ("compare", "versus", "ranking", "first", "second", "easier")):
                score += 0.45
        elif role == "skeptical_counterexample":
            if any(k in low for k in ("if", "unless", "counterexample", "falsif", "assumption")):
                score += 0.45
        elif role == "direct_answer":
            if any(k in low for k in ("easiest", "best", "pick", "choose", "recommend", "feasible", "verdict")):
                score += 0.35
        if len(re.findall(r"[a-z0-9']+", low)) >= 18:
            score += 0.1
        return max(0.0, min(1.0, score))

    def _candidate_quality(self, text: str) -> float:
        low = text.lower()
        stripped = " ".join(low.split())
        tokens = re.findall(r"[a-z0-9']+", low)
        if not tokens:
            return 0.0
        quality = 0.0
        n = len(tokens)
        if 18 <= n <= 120:
            quality += 0.45
        elif n >= 10:
            quality += 0.25
        if any(k in low for k in ("because", "therefore", "risk", "tradeoff", "mitigation", "step", "phase", "test")):
            quality += 0.30
        if any(k in low for k in ("based on the available context", "user asks", "user wants", "user proposed")):
            quality -= 0.35
        if any(k in low for k in ("strongest signals right now", "primary terms:", "additional angle")):
            quality -= 0.50
        if stripped.startswith("user "):
            quality -= 0.55
        return max(0.0, min(1.0, quality))
