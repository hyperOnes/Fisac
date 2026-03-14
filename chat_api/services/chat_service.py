from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import re
import traceback
import time
from typing import AsyncIterator, Iterable, Optional
from uuid import uuid4

from chat_api.models import ConversationRecord, MessageRecord, RunRecord
from chat_api.repository import ChatRepository
from chat_api.services.context_window import ContextWindowPolicy
from chat_api.services.conversational_composer import ComposeInput, ConversationalComposer
from chat_api.services.decision_trace import DecisionTraceStore
from chat_api.services.fiscal_text_bridge import FiscalTextBridge
from chat_api.services.gemini_client import GeminiClient, OutputQuality
from chat_api.services.jury_scorer import JuryScore, JuryScorer
from chat_api.services.openai_client import OpenAIClient
from chat_api.services.reasoning_orchestrator import ReasoningCandidate, ReasoningOrchestrator
from chat_api.services.rendering import render_tool_result
from chat_api.services.sse import SSEEvent
from chat_api.services.summary_service import SummaryService
from chat_api.services.tool_service import ToolService
from chat_api.services.truth_reasoner import TruthReasoner


@dataclass
class ConversationDTO:
    id: str
    title: str
    created_at: str
    updated_at: str
    gemini_enabled: bool
    last_message_preview: Optional[str] = None


@dataclass
class MessageDTO:
    id: str
    role: str
    content: str
    created_at: str
    status: str
    run_id: Optional[str] = None
    latency_ms: Optional[float] = None
    confidence: Optional[float] = None
    mse: Optional[float] = None
    generation_source: Optional[str] = None
    generation_attempts: Optional[int] = None
    quality_flags: Optional[str] = None


@dataclass
class _ContextProbeResult:
    ok: bool
    provider: str
    focus: str
    context: str = ""
    error: str | None = None


class ChatService:
    def __init__(
        self,
        repo: ChatRepository,
        bridge: FiscalTextBridge,
        context_policy: ContextWindowPolicy,
        summary_service: SummaryService,
        tool_service: ToolService | None = None,
        composer: ConversationalComposer | None = None,
        gemini_client: GeminiClient | None = None,
        openai_client: OpenAIClient | None = None,
        truth_reasoner: TruthReasoner | None = None,
        orchestrator: ReasoningOrchestrator | None = None,
        jury_scorer: JuryScorer | None = None,
        decision_trace_store: DecisionTraceStore | None = None,
    ) -> None:
        self.repo = repo
        self.bridge = bridge
        self.context_policy = context_policy
        self.summary_service = summary_service
        self.tool_service = tool_service
        self.composer = composer or ConversationalComposer()
        self.gemini_client = gemini_client
        self.openai_client = openai_client
        self.truth_reasoner = truth_reasoner or TruthReasoner()
        self.orchestrator = orchestrator or ReasoningOrchestrator(gemini_client=gemini_client, openai_client=openai_client)
        self.jury_scorer = jury_scorer or JuryScorer(
            learned_head_path=self.bridge.settings.jury_head_path,
            learned_heads_enabled=self.bridge.settings.learned_heads_enabled,
        )
        self.decision_trace_store = decision_trace_store or DecisionTraceStore()
        self._semantic_context_cache: dict[str, str] = {}
        self._semantic_context_order: deque[str] = deque()
        self._probe_stats: deque[tuple[float, int, int]] = deque(maxlen=2000)

    def create_conversation(self, title: str | None, gemini_enabled: bool = True) -> ConversationDTO:
        if self.bridge.settings.per_chat_hard_reset and self.bridge.settings.generation_backend == "liquid_native":
            self.bridge.hard_reset_between_chats()
        rec = self.repo.create_conversation(title=title, gemini_enabled=gemini_enabled)
        return self._conversation_to_dto(rec)

    def list_conversations(self, limit: int) -> list[ConversationDTO]:
        rows = self.repo.list_conversations(limit=limit)
        return [self._conversation_to_dto(r) for r in rows]

    def list_messages(self, conversation_id: str, limit: int, before: str | None) -> list[MessageDTO]:
        rows = self.repo.list_messages(conversation_id=conversation_id, limit=limit, before=before)
        return [self._message_to_dto(r) for r in rows]

    async def stream_reply(self, conversation_id: str, user_text: str, run_id: str) -> AsyncIterator[SSEEvent]:
        conv = self.repo.get_conversation(conversation_id)
        if conv is None:
            yield SSEEvent(
                event="error",
                data={"run_id": run_id, "code": "conversation_not_found", "message": "Conversation not found."},
            )
            return

        started_at_iso = self._now_iso()
        started_perf = time.perf_counter()
        assistant_message_id = str(uuid4())

        yield SSEEvent(event="ack", data={"run_id": run_id, "conversation_id": conversation_id})

        try:
            self.repo.create_message(
                conversation_id=conversation_id,
                role="user",
                content=user_text,
                status="complete",
                run_id=None,
            )

            self._refresh_summary_if_needed(conversation_id)
            summary_text = self.bridge.get_summary(conversation_id)
            recent_messages = self.repo.list_messages(
                conversation_id=conversation_id,
                limit=max(1, self.context_policy.keep_last_messages),
                before=None,
            )

            tool_meta = self._maybe_execute_tool(
                conversation_id=conversation_id,
                user_text=user_text,
                run_id=run_id,
            )
            if tool_meta is not None:
                tool_text, tool_conf_override = tool_meta
                bridge_base_input = f"{user_text}\n[tool]\n{tool_text}"
            else:
                tool_text = None
                tool_conf_override = None
                bridge_base_input = user_text

            bridge_input = self._build_bridge_input(conversation_id=conversation_id, user_text=bridge_base_input)
            bridge_result = self.bridge.step(user_text=bridge_input, state_id=conversation_id)
            context_error: str | None = None
            context_probes_total = 0
            context_probes_success = 0
            candidate_count = 0
            winner_index: int | None = None
            winner_score: float | None = None
            answer_mode: str | None = None
            echo_score: float | None = None
            coverage_score: float | None = None
            backend = self.bridge.settings.generation_backend
            if self.bridge.settings.pure_llm_generation:
                backend = "gemini_raw"
            generation_source: str = "liquid_native" if backend == "liquid_native" else "deterministic"
            generation_attempts = 1
            quality_flags: list[str] = []
            candidates: list[ReasoningCandidate] = []

            if tool_text is not None:
                assistant_base_text = tool_text
                generation_source = "deterministic"
                answer_mode = "tool_result"
            else:
                if backend == "liquid_native":
                    followup_mode = self._followup_mode(user_text)
                    external_context = ""
                    context_flags: list[str] = []
                    if self.bridge.settings.external_context_enabled and conv.gemini_enabled:
                        external_context, context_flags, context_error, context_probes_total, context_probes_success = await self._collect_external_context(
                            user_text=user_text,
                            summary_text=summary_text,
                            recent_messages=recent_messages,
                        )
                        quality_flags.extend(context_flags)
                    else:
                        external_context = self._local_context_fallback(
                            user_text=user_text,
                            summary_text=summary_text,
                            recent_messages=recent_messages,
                        )

                    reasoning_seed = ""
                    reasoner_input_context = external_context or self._local_context_fallback(
                        user_text=user_text,
                        summary_text=summary_text,
                        recent_messages=recent_messages,
                    )
                    reasoning = self.truth_reasoner.reason(
                        user_text=user_text,
                        context_hint=reasoner_input_context,
                        recent_messages=recent_messages,
                    )
                    if reasoning is not None and reasoning.text.strip():
                        reasoning_seed = reasoning.text
                        quality_flags.extend(reasoning.flags)

                    liquid_decode_text, decode_flags = self.bridge.generate_liquid_native_text(
                        conversation_id=conversation_id,
                        user_text=user_text,
                        vector_out=bridge_result.vector_out,
                        recent_messages=recent_messages,
                        context_hint=external_context,
                        reasoning_seed=reasoning_seed,
                    )
                    quality_flags.extend(decode_flags)
                    generation_source = "liquid_native"
                    orchestrator_flags: list[str] = []
                    if self.bridge.settings.jury_mode:
                        ext_candidates, orchestrator_flags = await self.orchestrator.generate_candidates(
                            user_text=user_text,
                            summary=summary_text,
                            recent_messages=recent_messages,
                            external_context=external_context,
                        )
                        candidates.extend(ext_candidates)
                        quality_flags.extend(orchestrator_flags)
                        if not ext_candidates:
                            local_candidates = self._build_local_role_candidates(
                                user_text=user_text,
                                context_hint=external_context,
                                reasoning_seed=reasoning_seed,
                                recent_messages=recent_messages,
                            )
                            if local_candidates:
                                candidates.extend(local_candidates)
                                quality_flags.append("local_candidate_expansion")
                    if (
                        reasoning_seed.strip()
                        and not self._is_meta_or_echo_candidate(reasoning_seed, user_text)
                        and not self._looks_like_recent_user_echo(reasoning_seed, recent_messages)
                    ):
                        candidates.append(
                            ReasoningCandidate(
                                role="truth_reasoner_seed",
                                text=reasoning_seed,
                                provider="fisac_truth_reasoner",
                                score_hint=0.12,
                            )
                        )
                    if (
                        liquid_decode_text.strip()
                        and not self._is_weak_candidate_text(liquid_decode_text)
                        and not self._is_meta_or_echo_candidate(liquid_decode_text, user_text)
                        and not self._looks_like_recent_user_echo(liquid_decode_text, recent_messages)
                    ):
                        candidates.append(
                            ReasoningCandidate(
                                role="liquid_decode",
                                text=liquid_decode_text,
                                provider="fisac_liquid_decode",
                                score_hint=0.08,
                            )
                        )
                    if not candidates:
                        emergency = self._emergency_liquid_text(
                            user_text=user_text,
                            context_hint=external_context,
                            reasoning_seed=reasoning_seed,
                            recent_messages=recent_messages,
                        )
                        if emergency:
                            candidates.append(
                                ReasoningCandidate(
                                    role="emergency_recovery",
                                    text=emergency,
                                    provider="fisac_emergency",
                                    score_hint=0.06,
                                )
                            )
                            quality_flags.append("emergency_candidate")
                    candidate_count = len(candidates)

                    if candidates and self.bridge.settings.jury_mode:
                        best, ranked = self.jury_scorer.choose_best(
                            user_text=user_text,
                            candidates=candidates,
                            recent_messages=recent_messages,
                            query_vec=bridge_result.vector_out,
                            encode_text=self.bridge.encode_text,
                        )
                        if best is not None:
                            selected_index, selected_score = self._select_turn_winner(
                                best=best,
                                ranked=ranked,
                                candidates=candidates,
                                recent_messages=recent_messages,
                                user_text=user_text,
                                followup_mode=followup_mode,
                            )
                            winner = candidates[selected_index]
                            winner_index = selected_index
                            winner_score = float(selected_score.total)
                            echo_score = float(selected_score.echo)
                            coverage_score = float(selected_score.coverage)
                            support = self._pick_support_candidate(
                                primary_index=selected_index,
                                primary_score=selected_score.total,
                                ranked=ranked,
                                candidates=candidates,
                                recent_messages=recent_messages,
                                user_text=user_text,
                                followup_mode=followup_mode,
                            )
                            if support is not None:
                                merged_winner = self._merge_candidate_texts(
                                    primary=winner.text,
                                    support=support.text,
                                )
                                assistant_base_text = self.bridge.synthesize_winner_text(
                                    winner_text=merged_winner,
                                    user_text=user_text,
                                    max_sentences=5,
                                ) or winner.text
                                if merged_winner != winner.text:
                                    answer_mode = "liquid_jury_support"
                                    quality_flags.append("jury_support_merge")
                                else:
                                    answer_mode = "liquid_jury"
                                    quality_flags.append("jury_support_skipped_redundant")
                            else:
                                assistant_base_text = self.bridge.synthesize_winner_text(
                                    winner_text=winner.text,
                                    user_text=user_text,
                                    max_sentences=4,
                                ) or winner.text
                                answer_mode = "liquid_jury"
                            self.decision_trace_store.put(
                                run_id,
                                {
                                    "run_id": run_id,
                                    "conversation_id": conversation_id,
                                    "winner_index": winner_index,
                                    "winner_provider": winner.provider,
                                    "winner_role": winner.role,
                                    "winner_score": winner_score,
                                    "followup_mode": followup_mode,
                                    "support_role": support.role if support is not None else None,
                                    "support_provider": support.provider if support is not None else None,
                                    "candidate_count": len(candidates),
                                    "ranked": [
                                        {
                                            "index": s.index,
                                            "score": s.total,
                                            "coverage": s.coverage,
                                            "echo": s.echo,
                                            "actionability": s.actionability,
                                            "coherence": s.coherence,
                                            "alignment": s.alignment,
                                            "evidence": s.evidence,
                                            "contradiction_penalty": s.contradiction_penalty,
                                            "meta_penalty": s.meta_penalty,
                                            "generic_penalty": s.generic_penalty,
                                            "provider": candidates[s.index].provider,
                                            "role": candidates[s.index].role,
                                            "preview": candidates[s.index].text[:280],
                                        }
                                                for s in ranked[:5]
                                    ],
                                },
                            )
                        else:
                            fallback = self._pick_non_meta_candidate(candidates=candidates, user_text=user_text)
                            if fallback is not None:
                                assistant_base_text = self.bridge.synthesize_winner_text(
                                    winner_text=fallback.text,
                                    user_text=user_text,
                                    max_sentences=4,
                                ) or fallback.text
                                answer_mode = "liquid_fallback_candidate"
                            elif (
                                liquid_decode_text.strip()
                                and not self._is_weak_candidate_text(liquid_decode_text)
                                and not self._is_meta_or_echo_candidate(liquid_decode_text, user_text)
                            ):
                                assistant_base_text = liquid_decode_text
                                answer_mode = "liquid_fallback_decode"
                            elif reasoning_seed.strip() and not self._is_meta_or_echo_candidate(reasoning_seed, user_text):
                                assistant_base_text = self.bridge.synthesize_winner_text(
                                    winner_text=reasoning_seed,
                                    user_text=user_text,
                                    max_sentences=4,
                                ) or reasoning_seed
                                answer_mode = "liquid_fallback_reasoning_seed"
                            else:
                                assistant_base_text = self._emergency_liquid_text(
                                    user_text=user_text,
                                    context_hint=external_context,
                                    reasoning_seed=reasoning_seed,
                                    recent_messages=recent_messages,
                                )
                                quality_flags.append("liquid_emergency_recovery")
                                answer_mode = "liquid_emergency_recovery"
                    else:
                        if candidates:
                            filtered_candidates = [
                                c
                                for c in candidates
                                if (not self._is_weak_candidate_text(c.text))
                                and (not self._is_meta_or_echo_candidate(c.text, user_text))
                            ]
                            if filtered_candidates:
                                candidates = filtered_candidates
                            # If jury mode is disabled, still select the strongest candidate by liquid alignment.
                            ranked_simple = sorted(
                                candidates,
                                key=lambda c: (
                                    c.score_hint
                                    + self.bridge.score_candidate_text(
                                        query_vec=bridge_result.vector_out,
                                        text=c.text,
                                    )
                                ),
                                reverse=True,
                            )
                            winner = ranked_simple[0]
                            assistant_base_text = self.bridge.synthesize_winner_text(
                                winner_text=winner.text,
                                user_text=user_text,
                                max_sentences=4,
                            ) or winner.text
                            answer_mode = "liquid_single_pick"
                        elif (
                            reasoning_seed.strip()
                            and not self._is_weak_candidate_text(reasoning_seed)
                            and not self._is_meta_or_echo_candidate(reasoning_seed, user_text)
                        ):
                            assistant_base_text = self.bridge.synthesize_winner_text(
                                winner_text=reasoning_seed,
                                user_text=user_text,
                                max_sentences=4,
                            ) or reasoning_seed
                            answer_mode = "liquid_reasoning_seed"
                        elif (
                            liquid_decode_text.strip()
                            and not self._is_weak_candidate_text(liquid_decode_text)
                            and not self._is_meta_or_echo_candidate(liquid_decode_text, user_text)
                        ):
                            assistant_base_text = liquid_decode_text
                            answer_mode = "liquid_decode_only"
                        else:
                            assistant_base_text = self._emergency_liquid_text(
                                user_text=user_text,
                                context_hint=external_context,
                                reasoning_seed=reasoning_seed,
                                recent_messages=recent_messages,
                            )
                            quality_flags.append("liquid_emergency_recovery")
                            answer_mode = "liquid_emergency_recovery"
                    if self._is_meta_or_echo_candidate(assistant_base_text, user_text):
                        quality_flags.append("meta_or_echo_candidate")
                        alt = self._pick_non_meta_candidate(candidates=candidates, user_text=user_text)
                        if alt is not None:
                            assistant_base_text = self.bridge.synthesize_winner_text(
                                winner_text=alt.text,
                                user_text=user_text,
                                max_sentences=4,
                            ) or alt.text
                            answer_mode = "liquid_meta_echo_replaced"
                    if self._looks_like_recent_user_echo(assistant_base_text, recent_messages):
                        quality_flags.append("history_user_echo")
                        alt = self._pick_non_meta_candidate(candidates=candidates, user_text=user_text)
                        if alt is not None and not self._looks_like_recent_user_echo(alt.text, recent_messages):
                            assistant_base_text = self.bridge.synthesize_winner_text(
                                winner_text=alt.text,
                                user_text=user_text,
                                max_sentences=4,
                            ) or alt.text
                            answer_mode = "liquid_history_echo_replaced"
                    if followup_mode == "continue" and self._is_redundant_with_recent_assistants(assistant_base_text, recent_messages):
                        alt_novel = self._pick_most_novel_candidate(
                            candidates=candidates,
                            user_text=user_text,
                            recent_messages=recent_messages,
                            current_text=assistant_base_text,
                        )
                        if alt_novel is not None:
                            assistant_base_text = self.bridge.synthesize_winner_text(
                                winner_text=alt_novel.text,
                                user_text=user_text,
                                max_sentences=4,
                            ) or alt_novel.text
                            quality_flags.append("continue_novelty_swap")
                            answer_mode = "liquid_continue_novelty_swap"
                    if candidates:
                        max_recent_sim = self._max_similarity_to_recent_assistants(assistant_base_text, recent_messages)
                        if max_recent_sim >= 0.88:
                            alt_global = self._pick_most_novel_candidate(
                                candidates=candidates,
                                user_text=user_text,
                                recent_messages=recent_messages,
                                current_text=assistant_base_text,
                            )
                            if alt_global is not None and self._text_similarity(assistant_base_text, alt_global.text) < 0.80:
                                assistant_base_text = self.bridge.synthesize_winner_text(
                                    winner_text=alt_global.text,
                                    user_text=user_text,
                                    max_sentences=4,
                                ) or alt_global.text
                                quality_flags.append("global_novelty_swap")
                                answer_mode = "liquid_global_novelty_swap"
                    if self._is_redundant_with_recent_assistants(assistant_base_text, recent_messages):
                        diversified = self._diversify_repeated_answer(
                            text=assistant_base_text,
                            user_text=user_text,
                            recent_messages=recent_messages,
                        )
                        if diversified and diversified.strip():
                            assistant_base_text = diversified
                            quality_flags.append("forced_diversify_repeat")
                            answer_mode = "liquid_forced_diversify"
                    repeat_streak = self._user_prompt_repeat_streak(user_text=user_text, recent_messages=recent_messages)
                    if repeat_streak >= 2:
                        assistant_base_text = self._inject_repeat_progress(
                            text=assistant_base_text,
                            user_text=user_text,
                            repeat_streak=repeat_streak,
                            variant_seed=(len(recent_messages) + sum(ord(ch) for ch in user_text)) % 997,
                        )
                        quality_flags.append("repeat_progress_injected")
                        answer_mode = "liquid_repeat_progress"
                    assistant_repeat = self._assistant_text_repeat_streak(
                        text=assistant_base_text,
                        recent_messages=recent_messages,
                    )
                    if assistant_repeat >= 1:
                        assistant_base_text = self._inject_repeat_progress(
                            text=assistant_base_text,
                            user_text=user_text,
                            repeat_streak=assistant_repeat + 1,
                            variant_seed=((len(recent_messages) * 31) + sum(ord(ch) for ch in user_text)) % 997,
                        )
                        quality_flags.append("assistant_repeat_breaker")
                        answer_mode = "liquid_repeat_breaker"
                    if self._is_weak_candidate_text(assistant_base_text):
                        quality_flags.append("weak_liquid_decode")
                elif backend == "gemini_raw":
                    if self.gemini_client is None or not self.gemini_client.configured:
                        assistant_base_text = "no better signal"
                        generation_source = "gemini_raw"
                        quality_flags.append("gemini_raw_unavailable")
                        answer_mode = "gemini_raw_unavailable"
                    else:
                        gemini_raw = await self.gemini_client.generate_raw(user_text=user_text, recent_messages=recent_messages)
                        if gemini_raw.ok:
                            assistant_base_text = self.composer.sanitize_generated(gemini_raw.text)
                            generation_source = "gemini_raw"
                            generation_attempts = gemini_raw.attempts
                            answer_mode = "gemini_raw"
                        else:
                            assistant_base_text = "no better signal"
                            generation_source = "gemini_raw"
                            generation_attempts = gemini_raw.attempts
                            context_error = gemini_raw.error
                            quality_flags.append("gemini_raw_error")
                            answer_mode = "gemini_raw_error"
                else:
                    deterministic = self.composer.compose(
                        ComposeInput(
                            user_text=user_text,
                            bridge_text=bridge_result.assistant_text,
                            summary=summary_text,
                            recent_messages=recent_messages,
                            tool_text=None,
                        )
                    )
                    assistant_base_text = deterministic
                    generation_source = "deterministic"
                    answer_mode = "deterministic_compose"

                if self.bridge.settings.response_guard_enabled:
                    quality = self._evaluate_text_quality(assistant_base_text)
                    quality_flags.extend(quality.flags)
                    if (
                        generation_source == "liquid_native"
                        and (not quality.is_valid)
                        and candidates
                        and generation_attempts < 2
                    ):
                        retry_candidate = self._pick_most_novel_candidate(
                            candidates=candidates,
                            user_text=user_text,
                            recent_messages=recent_messages,
                            current_text=assistant_base_text,
                        )
                        if retry_candidate is not None:
                            retry_text = self.bridge.synthesize_winner_text(
                                winner_text=retry_candidate.text,
                                user_text=user_text,
                                max_sentences=4,
                            ) or retry_candidate.text
                            if retry_text.strip() and (not self._is_meta_or_echo_candidate(retry_text, user_text)):
                                assistant_base_text = retry_text
                                generation_attempts = 2
                                quality_flags.append("liquid_guard_retry")
                                quality = self._evaluate_text_quality(assistant_base_text)
                                quality_flags.extend(quality.flags)
            assistant_base_text = self._ensure_not_exact_repeat(
                text=assistant_base_text,
                user_text=user_text,
                recent_messages=recent_messages,
            )
            assembled: list[str] = []
            for index, chunk in enumerate(self._chunk_text(assistant_base_text)):
                assembled.append(chunk)
                yield SSEEvent(event="token", data={"run_id": run_id, "delta": chunk, "index": index})

            assistant_text = "".join(assembled).strip()
            metrics = self.bridge.get_runtime_metrics(conversation_id)
            mse = float(metrics.get("mse", 0.0))
            confidence = float(metrics.get("confidence", bridge_result.confidence))
            if tool_conf_override is not None:
                confidence = tool_conf_override
            else:
                if generation_source == "liquid_native":
                    if "liquid_native_retry" in quality_flags:
                        confidence *= 0.95
                    if "prompt_echo" in quality_flags:
                        confidence *= 0.82
                    if "incomplete_tail" in quality_flags:
                        confidence *= 0.90
                if generation_source == "gemini_raw":
                    confidence *= 0.92
                if context_error and generation_source == "liquid_native":
                    confidence *= 0.97
                confidence = max(0.05, min(0.99, confidence))

            self.repo.create_message(
                conversation_id=conversation_id,
                role="assistant",
                content=assistant_text,
                status="complete",
                run_id=run_id,
                message_id=assistant_message_id,
            )

            self.bridge.teach(user_text=user_text, assistant_text=assistant_text, state_id=conversation_id)

            ended_at_iso = self._now_iso()
            latency_ms = float(metrics.get("latency_ms", bridge_result.latency_ms))
            quality_flags_clean = sorted({f for f in quality_flags if f})
            self.repo.create_run(
                RunRecord(
                    id=run_id,
                    conversation_id=conversation_id,
                    started_at=started_at_iso,
                    ended_at=ended_at_iso,
                    latency_ms=latency_ms,
                    mse=mse,
                    confidence=confidence,
                    pruned_now=int(metrics.get("pruned_now", bridge_result.pruned_now)),
                    myelinated_now=int(metrics.get("myelinated_now", bridge_result.myelinated_now)),
                    generation_source=generation_source,  # type: ignore[arg-type]
                    generation_attempts=max(1, generation_attempts),
                    quality_flags=",".join(quality_flags_clean) if quality_flags_clean else None,
                    output_chars=len(assistant_text),
                    runtime_profile=("jury_mode" if self.bridge.settings.jury_mode else self.bridge.settings.runtime_profile),
                    baseline_id=self.jury_scorer.head_version or self.bridge.settings.jury_head_version,
                    context_probes_total=int(context_probes_total),
                    context_probes_success=int(context_probes_success),
                    candidate_count=int(candidate_count),
                    winner_index=winner_index,
                    winner_score=winner_score,
                    answer_mode=answer_mode,
                    echo_score=echo_score,
                    coverage_score=coverage_score,
                    error_code="external_context_error" if context_error else None,
                    error_message=context_error,
                )
            )
            self.repo.touch_conversation(conversation_id)

            yield SSEEvent(
                event="done",
                data={
                    "run_id": run_id,
                    "assistant_message_id": assistant_message_id,
                    "latency_ms": round(latency_ms, 3),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            elapsed_ms = (time.perf_counter() - started_perf) * 1000.0
            err = str(exc).strip() or repr(exc)
            tb = traceback.format_exc().strip()
            error_text = f"Error: {err}"
            try:
                self.repo.create_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=error_text,
                    status="error",
                    run_id=run_id,
                    message_id=assistant_message_id,
                )
            except Exception:
                # If the assistant message already exists, mark it as error in-place.
                try:
                    self.repo.update_message(assistant_message_id, error_text, "error")
                except Exception:
                    pass
            try:
                self.repo.create_run(
                    RunRecord(
                        id=run_id,
                        conversation_id=conversation_id,
                        started_at=started_at_iso,
                        ended_at=self._now_iso(),
                        latency_ms=elapsed_ms,
                        mse=0.0,
                        confidence=0.0,
                        pruned_now=0,
                        myelinated_now=0,
                        generation_source="liquid_native"
                        if self.bridge.settings.generation_backend == "liquid_native"
                        and not self.bridge.settings.pure_llm_generation
                        else "deterministic",
                        generation_attempts=1,
                        quality_flags="chat_runtime_error",
                        output_chars=0,
                        runtime_profile=("jury_mode" if self.bridge.settings.jury_mode else self.bridge.settings.runtime_profile),
                        baseline_id=self.jury_scorer.head_version or self.bridge.settings.jury_head_version,
                        error_code="chat_runtime_error",
                        error_message=(f"{err}\n{tb}")[:4000],
                    )
                )
            except Exception:
                pass
            yield SSEEvent(
                event="error",
                data={"run_id": run_id, "code": "chat_runtime_error", "message": err},
            )

    async def run_lifecycle_maintenance(self) -> list[dict[str, object]]:
        events = self.bridge.run_global_lifecycle_if_due(force=False)
        now = self._now_iso()
        for event in events:
            self.repo.create_run(
                RunRecord(
                    id=str(uuid4()),
                    conversation_id=str(event["conversation_id"]),
                    started_at=now,
                    ended_at=now,
                    latency_ms=0.0,
                    mse=0.0,
                    confidence=0.0,
                    pruned_now=int(event.get("pruned_now", 0)),
                    myelinated_now=int(event.get("myelinated_now", 0)),
                    runtime_profile=("jury_mode" if self.bridge.settings.jury_mode else self.bridge.settings.runtime_profile),
                    baseline_id=self.jury_scorer.head_version or self.bridge.settings.jury_head_version,
                    error_code=None,
                    error_message=None,
                )
            )
        return events

    def update_conversation(
        self,
        conversation_id: str,
        title: str | None = None,
        gemini_enabled: bool | None = None,
    ) -> ConversationDTO | None:
        rec = self.repo.update_conversation(
            conversation_id=conversation_id,
            title=title,
            gemini_enabled=gemini_enabled,
        )
        if rec is None:
            return None
        return self._conversation_to_dto(rec)

    def delete_conversation(self, conversation_id: str) -> bool:
        return self.repo.delete_conversation(conversation_id)

    def shutdown(self) -> None:
        self.bridge.persist_all()

    def _refresh_summary_if_needed(self, conversation_id: str) -> None:
        user_turns = self.repo.count_user_messages(conversation_id)
        if not self.context_policy.should_refresh_summary(user_turns):
            return
        recent = self.repo.list_recent_messages_for_summary(conversation_id=conversation_id, limit=80)
        existing = self.bridge.get_summary(conversation_id)
        summary = self.summary_service.update_summary(recent, previous_summary=existing)
        self.bridge.set_summary(conversation_id, summary)

    def _maybe_execute_tool(self, conversation_id: str, user_text: str, run_id: str) -> tuple[str, float] | None:
        if self.tool_service is None:
            return None
        try:
            parsed = self.tool_service.parse_tool_command(user_text)
        except Exception as exc:
            return f"Tool command parse error: {exc}", 0.2
        if parsed is None:
            return None
        tool_name, tool_args = parsed
        if tool_name == "context.search" and "conversation_id" not in tool_args:
            tool_args = {**tool_args, "conversation_id": conversation_id}
        result = self.tool_service.call_tool(tool_name, tool_args)
        self.repo.create_tool_call(
            conversation_id=conversation_id,
            tool_name=tool_name,
            tool_args=tool_args,
            ok=result.ok,
            output=result.output,
            run_id=run_id,
            error=result.error,
        )
        response_text = render_tool_result(
            tool=result.tool,
            payload=result.output,
            ok=result.ok,
            error=result.error,
        )
        # Tool execution is deterministic and local; keep confidence high on success.
        return response_text, (0.95 if result.ok else 0.25)

    def _build_bridge_input(self, conversation_id: str, user_text: str) -> str:
        history = self.repo.list_messages(
            conversation_id=conversation_id,
            limit=max(1, self.context_policy.keep_last_messages),
            before=None,
        )
        context_text = self.context_policy.to_text(history)
        if not context_text.strip():
            return user_text
        return f"{user_text}\n[context]\n{context_text}"

    def _chunk_text(self, text: str) -> Iterable[str]:
        parts = re.findall(r"\S+\s*|\n", text)
        if not parts:
            return [text]
        return parts

    async def _collect_external_context(
        self,
        *,
        user_text: str,
        summary_text: str,
        recent_messages: list[MessageRecord],
    ) -> tuple[str, list[str], str | None, int, int]:
        providers: list[str] = []
        if self.gemini_client is not None and self.gemini_client.configured:
            providers.append("gemini")
        if self.openai_client is not None and self.openai_client.configured:
            providers.append("openai")

        probe_total = 0
        probe_success = 0

        if not providers:
            cached = self._read_cached_context(user_text=user_text, summary_text=summary_text)
            if cached:
                self._record_probe_stats(total=0, success=0)
                return cached, ["external_context_cached_fallback"], None, 0, 0
            local = self._local_context_fallback(
                user_text=user_text,
                summary_text=summary_text,
                recent_messages=recent_messages,
            )
            if local:
                self._record_probe_stats(total=0, success=0)
                return local, ["external_context_local_fallback"], None, 0, 0
            self._record_probe_stats(total=0, success=0)
            return "", ["external_context_unavailable"], "no_context_provider_configured", 0, 0

        probe_count = max(1, min(8, int(self.bridge.settings.context_query_count)))
        focuses = self._build_context_focuses(user_text=user_text, recent_messages=recent_messages, count=probe_count)
        short_prompt = len(re.findall(r"[A-Za-z0-9']+", user_text)) <= 3
        target_successes = min(2 if short_prompt else 3, probe_count)

        tasks = []
        for i, focus in enumerate(focuses):
            tasks.append(
                self._run_context_probe(
                    providers=providers,
                    preferred_provider=providers[i % len(providers)],
                    focus=focus,
                    user_text=user_text,
                    summary_text=summary_text,
                    recent_messages=recent_messages,
                )
            )
        results: list[_ContextProbeResult] = list(await asyncio.gather(*tasks))
        probe_total = len(results)

        context_chunks: list[str] = []
        errors: list[str] = []
        flags: list[str] = []
        provider_ok: dict[str, int] = {}
        provider_err: dict[str, int] = {}
        for result in results:
            if result.ok and result.context:
                probe_success += 1
                context_chunks.append(result.context)
                provider_ok[result.provider] = provider_ok.get(result.provider, 0) + 1
            else:
                provider_err[result.provider] = provider_err.get(result.provider, 0) + 1
                if result.error:
                    errors.append(f"{result.provider}:{result.error}")

        merged = self._merge_context_chunks(context_chunks, user_text=user_text)
        if merged:
            self._write_cached_context(user_text=user_text, summary_text=summary_text, context=merged)
            for provider, count in provider_ok.items():
                flags.append(f"context_{provider}_ok_{count}")
            flags.append("external_context_used")
            if probe_success < target_successes:
                flags.append("external_context_partial")
            self._record_probe_stats(total=probe_total, success=probe_success)
            return merged, sorted(set(flags)), None, probe_total, probe_success

        cached = self._read_cached_context(user_text=user_text, summary_text=summary_text)
        if cached:
            flags.append("external_context_cached_fallback")
            if errors:
                flags.append("external_context_partial")
            self._record_probe_stats(total=probe_total, success=probe_success)
            return cached, sorted(set(flags)), None, probe_total, probe_success

        local = self._local_context_fallback(
            user_text=user_text,
            summary_text=summary_text,
            recent_messages=recent_messages,
        )
        if local:
            flags.append("external_context_local_fallback")
            if errors:
                flags.append("external_context_partial")
            self._record_probe_stats(total=probe_total, success=probe_success)
            return local, sorted(set(flags)), None, probe_total, probe_success

        flags.append("external_context_error")
        err = errors[0] if errors else "external_context_error"
        self._record_probe_stats(total=probe_total, success=probe_success)
        return "", sorted(set(flags)), err, probe_total, probe_success

    async def _run_context_probe(
        self,
        *,
        providers: list[str],
        preferred_provider: str,
        focus: str,
        user_text: str,
        summary_text: str,
        recent_messages: list[MessageRecord],
    ) -> _ContextProbeResult:
        order = [preferred_provider] + [p for p in providers if p != preferred_provider]
        last_error: str | None = None

        for provider in order:
            if provider == "gemini":
                if self.gemini_client is None or not self.gemini_client.configured:
                    continue
                result = await self.gemini_client.extract_context(
                    user_text=user_text,
                    summary=summary_text,
                    recent_messages=recent_messages,
                    focus=focus,
                )
                if result.ok and result.context:
                    return _ContextProbeResult(ok=True, provider="gemini", focus=focus, context=result.context)
                last_error = result.error or "gemini_context_error"
                continue

            if provider == "openai":
                if self.openai_client is None or not self.openai_client.configured:
                    continue
                result = await self.openai_client.extract_context(
                    user_text=user_text,
                    summary=summary_text,
                    recent_messages=recent_messages,
                    focus=focus,
                )
                if result.ok and result.context:
                    return _ContextProbeResult(ok=True, provider="openai", focus=focus, context=result.context)
                last_error = result.error or "openai_context_error"

        return _ContextProbeResult(
            ok=False,
            provider=preferred_provider,
            focus=focus,
            context="",
            error=last_error or "context_probe_failed",
        )

    def _build_context_focuses(self, *, user_text: str, recent_messages: list[MessageRecord], count: int) -> list[str]:
        user_lower = user_text.lower()
        short_prompt = len(re.findall(r"[A-Za-z0-9']+", user_text)) <= 3
        last_assistant = ""
        for msg in reversed(recent_messages):
            if msg.role == "assistant" and msg.content.strip():
                last_assistant = msg.content.strip()
                break

        focuses = [
            "core intent and explicit ask",
            "hard constraints and non-negotiables",
            "risks failure modes and missing assumptions",
            "best alternative options with tradeoffs",
            "decision rule and immediate next step",
        ]
        if "why" in user_lower or "explain" in user_lower:
            focuses[3] = "causal explanation and strongest evidence path"
        if " or " in user_lower:
            focuses[4] = "side-by-side comparison and pick-one recommendation"
        if short_prompt and last_assistant:
            focuses[0] = "infer likely intent from previous assistant turn and user follow-up"

        while len(focuses) < count:
            focuses.append(f"additional angle {len(focuses) + 1}")
        return focuses[:count]

    def _merge_context_chunks(self, chunks: list[str], *, user_text: str) -> str:
        sentence_scores: dict[str, float] = {}
        sentence_order: dict[str, int] = {}
        sentence_text: dict[str, str] = {}
        user_tokens = set(re.findall(r"[A-Za-z0-9']+", user_text.lower()))
        order_counter = 0

        for chunk in chunks:
            local_seen: set[str] = set()
            for raw in re.split(r"(?<=[.!?])\s+|\n+", chunk):
                sentence = " ".join(raw.strip().split())
                if not sentence:
                    continue
                low = sentence.lower()
                if (
                    low.startswith("based on the available context")
                    or low.startswith("user asks")
                    or low.startswith("the user asks")
                    or low.startswith("user wants")
                    or low.startswith("user proposed")
                    or low.startswith("user demands")
                    or low.startswith("user requested")
                    or low.startswith("user request")
                    or low.startswith("previous assistant")
                    or low.startswith("previous user")
                ):
                    continue
                norm = self._norm_text(sentence)
                if not norm:
                    continue
                if norm in local_seen:
                    continue
                local_seen.add(norm)
                words = re.findall(r"[A-Za-z0-9']+", sentence.lower())
                if len(words) < 3:
                    continue
                overlap_with_user = len(set(words) & user_tokens) / max(1, len(set(words)))
                if overlap_with_user > 0.88 and len(user_tokens) >= 8:
                    continue
                if norm not in sentence_order:
                    sentence_order[norm] = order_counter
                    order_counter += 1
                    sentence_text[norm] = sentence
                overlap = len(set(words) & user_tokens) / max(1, len(set(words)))
                richness = min(1.0, len(words) / 12.0)
                sentence_scores[norm] = sentence_scores.get(norm, 0.0) + (1.0 + 0.9 * overlap + 0.3 * richness)

        if not sentence_scores:
            return ""

        ranked = sorted(sentence_scores.items(), key=lambda item: (-item[1], sentence_order.get(item[0], 0)))
        out: list[str] = []
        for norm, _ in ranked:
            sentence = sentence_text.get(norm, "")
            if not sentence:
                continue
            if sentence[-1] not in ".!?":
                sentence = sentence.rstrip(" ,;:-") + "."
            out.append(sentence)
            if len(out) >= 10:
                break
        if not out:
            return ""
        return " ".join(out)

    def _local_context_fallback(
        self,
        *,
        user_text: str,
        summary_text: str,
        recent_messages: list[MessageRecord],
    ) -> str:
        segments: list[str] = []
        user_lower = user_text.lower()
        key_terms = self._extract_content_terms(user_text, max_terms=8)
        if " or " in user_lower or " vs " in user_lower or " versus " in user_lower:
            segments.append("Decision target: choose one option by near-term feasibility, safety margin, and integration complexity.")
        if any(token in user_lower for token in ("how", "exactly", "steps", "plan", "solve", "implement", "build")):
            segments.append("User asks for concrete implementation mechanics with measurable pass/fail gates.")
        if summary_text.strip():
            summary = " ".join(summary_text.strip().split())
            segments.append("Recent summary: " + summary[:280].rstrip(" ,;:-") + ".")
        recent_user_facts: list[str] = []
        for msg in reversed(recent_messages):
            if msg.role != "user":
                continue
            for raw in re.split(r"(?<=[.!?])\s+|\n+", msg.content):
                sentence = " ".join(raw.strip().split())
                if not sentence:
                    continue
                if self._looks_like_prompt_echo(sentence, user_text):
                    continue
                if self._is_meta_or_echo_candidate(sentence, user_text):
                    continue
                recent_user_facts.append(sentence)
                if len(recent_user_facts) >= 2:
                    break
            if len(recent_user_facts) >= 2:
                break
        if recent_user_facts:
            segments.extend(recent_user_facts[:2])
        if not segments and key_terms:
            segments.append("Active technical terms include " + ", ".join(key_terms[:5]) + ".")
        if not segments:
            segments.append("No stable context signal yet; provide one measurable objective and one hard constraint.")
        return " ".join(segments).strip()

    def _build_local_role_candidates(
        self,
        *,
        user_text: str,
        context_hint: str,
        reasoning_seed: str,
        recent_messages: list[MessageRecord],
    ) -> list[ReasoningCandidate]:
        user_tokens = set(re.findall(r"[A-Za-z0-9']+", user_text.lower()))
        pool: list[str] = []
        for text in (reasoning_seed, context_hint):
            if text.strip():
                pool.extend(re.split(r"(?<=[.!?])\s+|\n+", text))
        for msg in recent_messages[-10:]:
            if msg.role != "user":
                continue
            pool.extend(re.split(r"(?<=[.!?])\s+|\n+", msg.content))

        scored: list[tuple[float, float, float, float, str]] = []
        for raw in pool:
            sentence = " ".join(raw.strip().split())
            if not sentence:
                continue
            if self._is_weak_candidate_text(sentence):
                continue
            if self._is_meta_or_echo_candidate(sentence, user_text):
                continue
            if self._looks_like_recent_user_echo(sentence, recent_messages):
                continue
            tokens = set(re.findall(r"[A-Za-z0-9']+", sentence.lower()))
            if len(tokens) < 5:
                continue
            if "?" in sentence:
                continue
            overlap = len(tokens & user_tokens) / max(1, len(user_tokens)) if user_tokens else 0.0
            if user_tokens and overlap < 0.06:
                continue
            impl_hits = sum(
                1 for tok in tokens if tok in {"step", "phase", "build", "test", "measure", "prototype", "deploy", "validate"}
            )
            risk_hits = sum(
                1 for tok in tokens if tok in {"risk", "failure", "jam", "unstable", "hazard", "abort", "safety", "drift"}
            )
            compare_hits = sum(
                1 for tok in tokens if tok in {"compare", "versus", "vs", "easier", "harder", "first", "second", "rank"}
            )
            direct_score = (1.8 * overlap) + (0.2 * min(1.0, (impl_hits + risk_hits + compare_hits) / 3.0))
            impl_score = (1.5 * overlap) + (0.5 * min(1.0, impl_hits / 2.0))
            risk_score = (1.5 * overlap) + (0.5 * min(1.0, risk_hits / 2.0))
            compare_score = (1.4 * overlap) + (0.6 * min(1.0, compare_hits / 2.0))
            scored.append((direct_score, impl_score, risk_score, compare_score, sentence))

        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
        role_rows: list[tuple[str, float, str]] = []
        if scored:
            role_rows.append(("direct_answer", scored[0][0], scored[0][4]))
            best_impl = max(scored, key=lambda item: item[1])
            best_risk = max(scored, key=lambda item: item[2])
            best_compare = max(scored, key=lambda item: item[3])
            if best_impl[1] >= 0.18:
                role_rows.append(("implementation_sequence", best_impl[1], best_impl[4]))
            if best_risk[2] >= 0.18:
                role_rows.append(("failure_modes", best_risk[2], best_risk[4]))
            if best_compare[3] >= 0.18:
                role_rows.append(("comparative_feasibility", best_compare[3], best_compare[4]))

        if not role_rows:
            terms = self._extract_content_terms(user_text, max_terms=6)
            compare_request = any(tok in user_text.lower() for tok in (" or ", " vs ", " versus ", "which", "easiest", "compare"))
            if len(terms) >= 2:
                t0, t1 = terms[0], terms[1]
                role_rows = [
                    ("direct_answer", 0.22, f"Prioritize the path with lower {t0}-{t1} coupling and fewer safety-critical moving parts in phase one."),
                    ("implementation_sequence", 0.22, f"Run bench validation first, then low-speed closed-loop tests, then capped speed ramps while measuring {t0} and {t1}."),
                    ("failure_modes", 0.22, f"Primary failure mode is unstable {t0}-{t1} coupling at speed; mitigate with redundancy and hard abort gates."),
                ]
                if compare_request:
                    role_rows.append(
                        ("comparative_feasibility", 0.23, f"If forced to pick now, choose the option with lower {t0}-{t1} coupling and faster path to safety validation.")
                    )
            else:
                role_rows = [
                    ("direct_answer", 0.20, "Choose the option with lower integration complexity and a tighter safety envelope for first deployment."),
                    ("implementation_sequence", 0.20, "Run phased validation: bench subsystem tests, then low-speed integration, then capped speed ramps."),
                    ("failure_modes", 0.20, "Top risks are coupling instability, fault propagation, and recovery failure under load transitions."),
                ]
                if compare_request:
                    role_rows.append(
                        ("comparative_feasibility", 0.21, "If forced to pick now, choose the path with fewer moving parts and tighter safety margins.")
                    )

        out: list[ReasoningCandidate] = []
        seen: set[str] = set()
        for role, score, text in role_rows:
            clean = " ".join(text.strip().split())
            if not clean:
                continue
            norm = self._norm_text(clean)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append(
                ReasoningCandidate(
                    role=role,
                    text=clean,
                    provider="fisac_local",
                    score_hint=float(max(0.05, min(0.45, score))),
                )
            )
        return out

    def _emergency_liquid_text(
        self,
        *,
        user_text: str,
        context_hint: str,
        reasoning_seed: str,
        recent_messages: list[MessageRecord],
    ) -> str:
        # Final bounded recovery path: construct a concise, non-meta answer from best local signal.
        user_tokens = set(re.findall(r"[A-Za-z0-9']+", user_text.lower()))
        pool: list[str] = []
        if reasoning_seed.strip():
            pool.extend(re.split(r"(?<=[.!?])\s+|\n+", reasoning_seed))
        if context_hint.strip():
            pool.extend(re.split(r"(?<=[.!?])\s+|\n+", context_hint))
        for msg in reversed(recent_messages):
            if msg.role != "assistant":
                continue
            pool.extend(re.split(r"(?<=[.!?])\s+|\n+", msg.content))
            if len(pool) >= 40:
                break

        scored: list[tuple[float, str]] = []
        for raw in pool:
            sentence = " ".join(raw.strip().split())
            if not sentence:
                continue
            if self._is_weak_candidate_text(sentence):
                continue
            if self._is_meta_or_echo_candidate(sentence, user_text):
                continue
            tokens = set(re.findall(r"[A-Za-z0-9']+", sentence.lower()))
            if len(tokens) < 4:
                continue
            overlap = len(tokens & user_tokens) / max(1, len(user_tokens)) if user_tokens else 0.0
            action_hits = sum(
                1
                for token in tokens
                if token
                in {
                    "build",
                    "test",
                    "measure",
                    "compare",
                    "prototype",
                    "validate",
                    "risk",
                    "failure",
                    "mitigation",
                    "phase",
                    "step",
                }
            )
            score = (1.5 * overlap) + (0.25 * min(1.0, action_hits / 3.0))
            scored.append((score, sentence))

        scored.sort(key=lambda item: item[0], reverse=True)
        picked: list[str] = []
        seen: set[str] = set()
        for _, sentence in scored:
            norm = re.sub(r"[^a-z0-9']+", " ", sentence.lower()).strip()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            if sentence[-1] not in ".!?":
                sentence = sentence.rstrip(" ,;:-") + "."
            picked.append(sentence)
            if len(picked) >= 2:
                break
        if picked:
            return " ".join(picked)

        key_terms = self._extract_content_terms(user_text, max_terms=6)
        if len(key_terms) >= 2:
            return (
                f"Highest-confidence move is to test {key_terms[0]} against {key_terms[1]} under a measurable limit. "
                f"Then compare outcomes and keep the lower-risk path."
            )
        return (
            "Current evidence is insufficient for a high-confidence verdict. "
            "State one measurable target and one hard limit, and I will compute a concrete next step."
        )

    def _diversify_repeated_answer(
        self,
        *,
        text: str,
        user_text: str,
        recent_messages: list[MessageRecord],
    ) -> str:
        clean = " ".join(text.strip().split())
        if not clean:
            return ""
        terms = self._extract_content_terms(user_text, max_terms=8)
        if len(terms) < 2:
            for msg in reversed(recent_messages):
                if msg.role != "user":
                    continue
                terms = self._extract_content_terms(msg.content, max_terms=8)
                if len(terms) >= 2:
                    break
        if len(terms) >= 2:
            additions = (
                f"Next discriminating test: hold {terms[0]} fixed and vary {terms[1]} under the same safety threshold.",
                f"Decision rule: choose the option that minimizes {terms[0]}-coupled failure while preserving {terms[1]}.",
                f"Verification step: run an A/B trial focused on {terms[0]} and {terms[1]} before scaling.",
            )
        else:
            additions = (
                "Next discriminating test: isolate one variable and keep all other constraints fixed.",
                "Decision rule: keep only the path that passes a measurable safety threshold.",
                "Verification step: run one controlled A/B trial before further scaling.",
            )
        idx = len(recent_messages) % len(additions)
        sentence = additions[idx]
        merged = f"{clean} {sentence}".strip()
        return merged

    def _user_prompt_repeat_streak(self, *, user_text: str, recent_messages: list[MessageRecord]) -> int:
        target = " ".join(user_text.strip().lower().split())
        if not target:
            return 0
        streak = 0
        for msg in reversed(recent_messages):
            if msg.role != "user":
                continue
            norm = " ".join(msg.content.strip().lower().split())
            if not norm:
                continue
            if norm == target:
                streak += 1
            else:
                break
        return streak

    def _assistant_text_repeat_streak(self, *, text: str, recent_messages: list[MessageRecord]) -> int:
        target = " ".join(text.strip().lower().split())
        if not target:
            return 0
        streak = 0
        for msg in reversed(recent_messages):
            if msg.role != "assistant":
                continue
            norm = " ".join(msg.content.strip().lower().split())
            if not norm:
                continue
            if norm == target:
                streak += 1
            else:
                break
        return streak

    def _inject_repeat_progress(self, *, text: str, user_text: str, repeat_streak: int, variant_seed: int = 0) -> str:
        clean = " ".join(text.strip().split())
        if not clean:
            clean = self._emergency_liquid_text(
                user_text=user_text,
                context_hint="",
                reasoning_seed="",
                recent_messages=[],
            )
        terms = self._extract_content_terms(user_text, max_terms=6)
        if len(terms) >= 2:
            additions = (
                f"Iteration step: isolate {terms[0]} and quantify failure onset against {terms[1]}.",
                f"Iteration step: run a constrained A/B test on {terms[0]} while holding {terms[1]} constant.",
                f"Iteration step: define abort thresholds for {terms[0]} and {terms[1]} before scaling speed.",
                f"Iteration step: record recovery time when {terms[0]} exceeds limits under {terms[1]} load.",
            )
        else:
            additions = (
                "Iteration step: isolate one variable and measure where failure begins.",
                "Iteration step: run one constrained A/B test before scaling.",
                "Iteration step: define explicit abort thresholds before the next trial.",
                "Iteration step: measure recovery time under controlled fault injection.",
            )
        base_idx = (repeat_streak - 2 + max(0, int(variant_seed))) % len(additions)
        marker = additions[base_idx]
        if marker.lower() in clean.lower():
            for offset in range(1, len(additions)):
                alt = additions[(base_idx + offset) % len(additions)]
                if alt.lower() not in clean.lower():
                    marker = alt
                    break
            else:
                marker = "Iteration step: introduce one new controlled variable and rerun the same pass/fail gate."
        merged = f"{clean} {marker}".strip()
        return merged

    def _extract_content_terms(self, text: str, *, max_terms: int) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
        stop = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "if", "in", "is",
            "it", "its", "of", "on", "or", "that", "the", "to", "we", "what", "when", "where", "which",
            "why", "with", "you", "your",
        }
        out: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if len(token) < 3 or token in stop:
                continue
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
            if len(out) >= max_terms:
                break
        return out

    def _ensure_not_exact_repeat(self, *, text: str, user_text: str, recent_messages: list[MessageRecord]) -> str:
        clean = " ".join(text.strip().split())
        if not clean:
            return clean
        last = ""
        for msg in reversed(recent_messages):
            if msg.role == "assistant":
                last = " ".join(msg.content.strip().split())
                break
        if not last:
            return clean
        if clean.lower() != last.lower():
            return clean
        terms = self._extract_content_terms(user_text, max_terms=4)
        variants = [
            "Next step: run one constrained prototype and keep strict abort thresholds.",
            "Next step: compare two options under the same pass/fail gate before scaling.",
            "Next step: isolate the highest-risk variable and measure failure onset directly.",
            "Next step: define a hard decision rule and reject paths that fail it early.",
        ]
        if len(terms) >= 2:
            variants.append(
                f"Next step: hold {terms[0]} constant and vary {terms[1]} to locate the break point."
            )
        idx = (len(recent_messages) + sum(ord(ch) for ch in user_text)) % len(variants)
        return f"{clean} {variants[idx]}".strip()

    def _merge_candidate_texts(self, *, primary: str, support: str) -> str:
        p_clean = " ".join(primary.strip().split())
        s_clean = " ".join(support.strip().split())
        if not p_clean:
            return s_clean
        if not s_clean:
            return p_clean
        if self._text_similarity(p_clean, s_clean) >= 0.70:
            return p_clean
        merged: list[str] = []
        seen: set[str] = set()
        for raw in re.split(r"(?<=[.!?])\s+|\n+", f"{p_clean} {s_clean}"):
            sentence = " ".join(raw.strip().split())
            if not sentence:
                continue
            if self._is_weak_candidate_text(sentence):
                continue
            norm = self._norm_text(sentence)
            if not norm or norm in seen:
                continue
            if any(self._text_similarity(sentence, prev) > 0.86 for prev in merged):
                continue
            seen.add(norm)
            merged.append(sentence)
            if len(merged) >= 5:
                break
        return " ".join(merged) if merged else p_clean

    def get_runtime_status(self) -> dict[str, object]:
        return {
            "jury_mode": bool(self.bridge.settings.jury_mode and self.bridge.settings.generation_backend == "liquid_native"),
            "context_probe_success_rate_1h": self._context_probe_success_rate_1h(),
            "provider_health": {
                "gemini": self.gemini_client.health() if self.gemini_client is not None else {"configured": False},
                "openai": self.openai_client.health() if self.openai_client is not None else {"configured": False},
            },
            "jury_head_version": self.jury_scorer.head_version or self.bridge.settings.jury_head_version,
            "synthesis_head_version": self.bridge.synthesis_head_version() or self.bridge.settings.synthesis_head_version,
            "learned_heads_enabled": bool(self.bridge.settings.learned_heads_enabled),
        }

    def get_decision_trace(self, run_id: str) -> dict[str, object] | None:
        return self.decision_trace_store.get(run_id)

    def _cache_key(self, user_text: str, summary_text: str = "") -> str:
        toks = re.findall(r"[A-Za-z0-9']+", user_text.lower())
        base = " ".join(toks[:24]).strip()
        summary_tokens = re.findall(r"[A-Za-z0-9']+", summary_text.lower())
        summary_norm = " ".join(summary_tokens[:24]).strip()
        if summary_norm:
            return f"{base}||{summary_norm}"
        return base

    def _write_cached_context(self, *, user_text: str, summary_text: str, context: str) -> None:
        key = self._cache_key(user_text, summary_text)
        if not key or not context.strip():
            return
        if key not in self._semantic_context_cache:
            self._semantic_context_order.append(key)
        self._semantic_context_cache[key] = context
        while len(self._semantic_context_order) > 500:
            old = self._semantic_context_order.popleft()
            self._semantic_context_cache.pop(old, None)

    def _read_cached_context(self, *, user_text: str, summary_text: str) -> str:
        key = self._cache_key(user_text, summary_text)
        if not key:
            return ""
        return self._semantic_context_cache.get(key, "")

    def _record_probe_stats(self, *, total: int, success: int) -> None:
        self._probe_stats.append((time.time(), int(total), int(success)))

    def _context_probe_success_rate_1h(self) -> float:
        now = time.time()
        cutoff = now - 3600.0
        total = 0
        success = 0
        while self._probe_stats and self._probe_stats[0][0] < cutoff:
            self._probe_stats.popleft()
        for _, t, s in self._probe_stats:
            total += max(0, t)
            success += max(0, s)
        if total <= 0:
            return 1.0
        return max(0.0, min(1.0, success / total))

    def _norm_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9']+", " ", text.lower()).strip()

    def _conversation_to_dto(self, rec: ConversationRecord) -> ConversationDTO:
        return ConversationDTO(
            id=rec.id,
            title=rec.title,
            created_at=rec.created_at,
            updated_at=rec.updated_at,
            gemini_enabled=rec.gemini_enabled,
            last_message_preview=rec.last_message_preview,
        )

    def _message_to_dto(self, rec: MessageRecord) -> MessageDTO:
        return MessageDTO(
            id=rec.id,
            role=rec.role,
            content=rec.content,
            created_at=rec.created_at,
            status=rec.status,
            run_id=rec.run_id,
            latency_ms=rec.latency_ms,
            confidence=rec.confidence,
            mse=rec.mse,
            generation_source=rec.generation_source,
            generation_attempts=rec.generation_attempts,
            quality_flags=rec.quality_flags,
        )

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _evaluate_text_quality(self, text: str) -> OutputQuality:
        if self.gemini_client is not None and hasattr(self.gemini_client, "evaluate_output"):
            quality = self.gemini_client.evaluate_output(text)  # type: ignore[no-any-return]
            filtered = [f for f in quality.flags if f not in {"too_short", "too_few_sentences"}]
            return OutputQuality(
                word_count=quality.word_count,
                sentence_count=quality.sentence_count,
                ends_cleanly=quality.ends_cleanly,
                has_repetition=quality.has_repetition,
                is_valid=not filtered,
                flags=filtered,
            )

        words = re.findall(r"[A-Za-z0-9']+", text)
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", text))
        ends_cleanly = bool(text.strip()) and text.strip()[-1] in ".!?"
        flags: list[str] = []
        if self.bridge.settings.response_incomplete_tail_guard and not ends_cleanly:
            flags.append("incomplete_tail")
        return OutputQuality(
            word_count=len(words),
            sentence_count=sentence_count,
            ends_cleanly=ends_cleanly,
            has_repetition=False,
            is_valid=not flags,
            flags=flags,
        )

    def _is_weak_candidate_text(self, text: str) -> bool:
        low = " ".join(text.strip().lower().split())
        if not low:
            return True
        if low == "no better signal":
            return True
        weak_patterns = (
            "strongest signals right now",
            "prioritize testing",
            "no better signal",
            "additional angle",
            "based on the available context",
            "the user asks",
            "user asks",
            "user wants",
            "user proposed",
            "user demands",
            "need concrete implementation steps with measurable checkpoints",
            "intent is underspecified",
            "core concepts involve",
            "primary terms:",
            "decision target:",
            "user asks for concrete implementation mechanics",
            "active technical terms include",
            "recent summary:",
        )
        return any(pattern in low for pattern in weak_patterns)

    def _followup_mode(self, user_text: str) -> str:
        low = " ".join(user_text.lower().split())
        if not low:
            return "none"
        if any(k in low for k in ("continue", "next", "then", "more detail", "elaborate")):
            return "continue"
        if any(k in low for k in ("how", "exactly", "steps", "plan", "solve", "implement", "build")):
            return "implementation"
        if any(k in low for k in ("why", "explain", "because")):
            return "explanation"
        if any(k in low for k in ("compare", "which", "easiest", "or", "vs", "versus")):
            return "comparison"
        return "none"

    def _preferred_roles_for_mode(self, mode: str) -> tuple[str, ...]:
        if mode == "continue":
            return ("implementation_sequence", "failure_modes", "direct_answer")
        if mode == "implementation":
            return ("implementation_sequence", "failure_modes", "direct_answer")
        if mode == "explanation":
            return ("failure_modes", "skeptical_counterexample", "comparative_feasibility", "direct_answer")
        if mode == "comparison":
            return ("comparative_feasibility", "direct_answer", "failure_modes")
        return ()

    def _support_roles_for_mode(self, mode: str) -> tuple[str, ...]:
        if mode in {"continue", "implementation"}:
            return ("failure_modes", "skeptical_counterexample", "comparative_feasibility")
        if mode == "explanation":
            return ("skeptical_counterexample", "failure_modes", "comparative_feasibility")
        if mode == "comparison":
            return ("failure_modes", "implementation_sequence")
        return ("failure_modes", "comparative_feasibility")

    def _select_turn_winner(
        self,
        *,
        best: JuryScore,
        ranked: list[JuryScore],
        candidates: list[ReasoningCandidate],
        recent_messages: list[MessageRecord],
        user_text: str,
        followup_mode: str,
    ) -> tuple[int, JuryScore]:
        preferred = set(self._preferred_roles_for_mode(followup_mode))
        scored: list[tuple[float, JuryScore]] = []
        history = self._recent_assistant_texts(recent_messages, limit=6)
        for score in ranked:
            cand = candidates[score.index]
            if self._is_meta_or_echo_candidate(cand.text, user_text):
                continue
            role_boost = 0.0
            if preferred:
                if cand.role in preferred:
                    role_boost += 0.12
                elif cand.role in {"truth_reasoner_seed", "liquid_decode"} and score.coverage >= 0.12 and score.evidence >= 0.30:
                    role_boost += 0.04
                else:
                    role_boost -= 0.04
            redundancy_penalty = 0.0
            max_sim, avg_sim = self._similarity_against_history(cand.text, history)
            if followup_mode == "continue":
                redundancy_penalty += (1.20 * max_sim) + (0.45 * avg_sim)
            elif followup_mode != "none":
                redundancy_penalty += (0.48 * max_sim) + (0.22 * avg_sim)
            else:
                redundancy_penalty += (0.20 * max_sim) + (0.08 * avg_sim)
            if self._is_redundant_with_recent_assistants(cand.text, recent_messages):
                if followup_mode == "continue":
                    redundancy_penalty += 0.80
                else:
                    redundancy_penalty += 0.22 if followup_mode != "none" else 0.12
            if (
                followup_mode in {"continue", "implementation", "explanation"}
                and max_sim >= 0.86
                and score.coverage < 0.82
            ):
                continue
            adjusted = float(score.total + role_boost - redundancy_penalty)
            scored.append((adjusted, score))
        if not scored:
            return best.index, best
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[0][1]
        # Keep strong external worker wins in near-tie regimes.
        external_preferred: JuryScore | None = None
        for _, candidate_score in scored:
            provider = candidates[candidate_score.index].provider
            if provider.startswith("fisac_"):
                continue
            external_preferred = candidate_score
            break
        if external_preferred is not None and (external_preferred.total + 0.12) >= selected.total:
            selected = external_preferred
        # Prefer richer evidence in comparison/explanation turns when score gap is small.
        selected_provider = candidates[selected.index].provider
        if followup_mode in {"comparison", "explanation"} and selected_provider.startswith("fisac_"):
            richer: JuryScore | None = None
            for _, candidate_score in scored[:4]:
                if candidate_score.evidence < 0.45:
                    continue
                if candidate_score.coverage + 0.02 < selected.coverage:
                    continue
                if candidate_score.total + 0.22 < selected.total:
                    continue
                if richer is None or candidate_score.evidence > richer.evidence:
                    richer = candidate_score
            if richer is not None:
                selected = richer
        if selected.index != best.index:
            if (best.total - selected.total) <= 0.70:
                return selected.index, selected
        return best.index, best

    def _pick_support_candidate(
        self,
        *,
        primary_index: int,
        primary_score: float,
        ranked: list[JuryScore],
        candidates: list[ReasoningCandidate],
        recent_messages: list[MessageRecord],
        user_text: str,
        followup_mode: str,
    ) -> ReasoningCandidate | None:
        primary = candidates[primary_index]
        support_roles = set(self._support_roles_for_mode(followup_mode))
        primary_text = primary.text
        history = self._recent_assistant_texts(recent_messages, limit=6)
        primary_tokens = set(re.findall(r"[A-Za-z0-9']+", primary_text.lower()))
        best_support: tuple[float, ReasoningCandidate] | None = None
        for score in ranked:
            if score.index == primary_index:
                continue
            if (primary_score - score.total) > 0.75:
                continue
            cand = candidates[score.index]
            if cand.role == primary.role:
                continue
            if self._is_meta_or_echo_candidate(cand.text, user_text):
                continue
            if self._is_redundant_with_recent_assistants(cand.text, recent_messages):
                continue
            sim_cutoff = 0.65 if followup_mode == "continue" else 0.82
            if self._text_similarity(primary_text, cand.text) >= sim_cutoff:
                continue
            cand_tokens = set(re.findall(r"[A-Za-z0-9']+", cand.text.lower()))
            token_overlap = len(primary_tokens & cand_tokens) / max(1, len(primary_tokens | cand_tokens))
            if token_overlap >= 0.72:
                continue
            history_max, _ = self._similarity_against_history(cand.text, history)
            if history_max >= (0.76 if followup_mode == "continue" else 0.88):
                continue
            role_boost = 0.16 if cand.role in support_roles else 0.0
            merged_score = float(score.total + role_boost)
            if best_support is None or merged_score > best_support[0]:
                best_support = (merged_score, cand)
        if best_support is None:
            return None
        return best_support[1]

    def _pick_most_novel_candidate(
        self,
        *,
        candidates: list[ReasoningCandidate],
        user_text: str,
        recent_messages: list[MessageRecord],
        current_text: str,
    ) -> ReasoningCandidate | None:
        last_assistant = ""
        for msg in reversed(recent_messages):
            if msg.role == "assistant" and msg.content.strip():
                last_assistant = msg.content
                break
        history = self._recent_assistant_texts(recent_messages, limit=6)
        baseline = current_text if current_text.strip() else last_assistant
        best: tuple[float, ReasoningCandidate] | None = None
        for cand in candidates:
            if self._is_weak_candidate_text(cand.text):
                continue
            if self._is_meta_or_echo_candidate(cand.text, user_text):
                continue
            if self._is_redundant_with_recent_assistants(cand.text, recent_messages):
                continue
            novelty_baseline = 1.0 - self._text_similarity(cand.text, baseline)
            history_max, history_avg = self._similarity_against_history(cand.text, history)
            novelty_history = max(0.0, 1.0 - ((0.7 * history_max) + (0.3 * history_avg)))
            novelty = (0.55 * novelty_baseline) + (0.45 * novelty_history)
            score = novelty + (0.2 * cand.score_hint)
            if best is None or score > best[0]:
                best = (score, cand)
        if best is None:
            return None
        return best[1]

    def _text_similarity(self, a: str, b: str) -> float:
        a_tokens = set(re.findall(r"[A-Za-z0-9']+", a.lower()))
        b_tokens = set(re.findall(r"[A-Za-z0-9']+", b.lower()))
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))

    def _is_meta_or_echo_candidate(self, text: str, user_text: str) -> bool:
        low = " ".join(text.strip().lower().split())
        if not low:
            return True
        if self._is_weak_candidate_text(low):
            return True
        meta_prefixes = (
            "based on the available context",
            "user asks",
            "the user asks",
            "user query",
            "user proposed",
            "user wants",
            "user demands",
            "previous user",
            "previous assistant",
            "working summary",
            "active topic terms",
        )
        if any(low.startswith(prefix) for prefix in meta_prefixes):
            return True
        return self._looks_like_prompt_echo(text, user_text)

    def _looks_like_prompt_echo(self, output_text: str, user_text: str) -> bool:
        out_words = re.findall(r"[A-Za-z0-9']+", output_text.lower())
        in_words = re.findall(r"[A-Za-z0-9']+", user_text.lower())
        if not out_words or not in_words:
            return False
        if len(in_words) <= 4:
            return False
        out_set = set(out_words)
        in_set = set(in_words)
        overlap = len(out_set & in_set) / max(1, len(out_set | in_set))
        starts = " ".join(out_words[: min(10, len(out_words))]) == " ".join(in_words[: min(10, len(in_words))])
        containment = " ".join(in_words[: min(14, len(in_words))]) in " ".join(out_words)
        if starts and overlap >= 0.42:
            return True
        if containment and overlap >= 0.35:
            return True
        return overlap >= 0.68

    def _looks_like_recent_user_echo(self, text: str, recent_messages: list[MessageRecord]) -> bool:
        candidate = " ".join(text.strip().lower().split())
        if not candidate:
            return False
        cand_tokens = re.findall(r"[A-Za-z0-9']+", candidate)
        if len(cand_tokens) < 6:
            return False
        for msg in reversed(recent_messages):
            if msg.role != "user":
                continue
            user_norm = " ".join(msg.content.strip().lower().split())
            if not user_norm:
                continue
            if self._looks_like_prompt_echo(candidate, user_norm):
                return True
            user_tokens = re.findall(r"[A-Za-z0-9']+", user_norm)
            if len(user_tokens) < 6:
                continue
            overlap = self._text_similarity(candidate, user_norm)
            if overlap >= 0.72:
                return True
        return False

    def _pick_non_meta_candidate(
        self,
        *,
        candidates: list[ReasoningCandidate],
        user_text: str,
    ) -> ReasoningCandidate | None:
        viable = [
            c
            for c in candidates
            if c.text.strip()
            and (not self._is_weak_candidate_text(c.text))
            and (not self._is_meta_or_echo_candidate(c.text, user_text))
        ]
        if not viable:
            return None
        viable.sort(key=lambda c: c.score_hint, reverse=True)
        return viable[0]

    def _recent_assistant_texts(self, recent_messages: list[MessageRecord], limit: int = 6) -> list[str]:
        out: list[str] = []
        for msg in reversed(recent_messages):
            if msg.role != "assistant":
                continue
            clean = " ".join(msg.content.strip().lower().split())
            if not clean:
                continue
            out.append(clean)
            if len(out) >= max(1, limit):
                break
        return out

    def _similarity_against_history(self, text: str, history: list[str]) -> tuple[float, float]:
        if not history:
            return 0.0, 0.0
        clean = " ".join(text.strip().lower().split())
        if not clean:
            return 0.0, 0.0
        sims = [self._text_similarity(clean, h) for h in history if h]
        if not sims:
            return 0.0, 0.0
        return max(sims), sum(sims) / len(sims)

    def _max_similarity_to_recent_assistants(self, text: str, recent_messages: list[MessageRecord]) -> float:
        history = self._recent_assistant_texts(recent_messages, limit=6)
        max_sim, _ = self._similarity_against_history(text, history)
        return max_sim

    def _is_redundant_with_recent_assistants(self, text: str, recent_messages: list[MessageRecord]) -> bool:
        candidate = " ".join(text.strip().lower().split())
        if not candidate:
            return False
        history = self._recent_assistant_texts(recent_messages, limit=6)
        if not history:
            return False
        if candidate in history:
            return True
        cand_tokens = set(re.findall(r"[A-Za-z0-9']+", candidate))
        if not cand_tokens:
            return False
        best_overlap = 0.0
        for prev in history:
            prev_tokens = set(re.findall(r"[A-Za-z0-9']+", prev))
            if not prev_tokens:
                continue
            overlap = len(cand_tokens & prev_tokens) / max(1, len(cand_tokens | prev_tokens))
            if overlap > best_overlap:
                best_overlap = overlap
        return best_overlap >= 0.84
