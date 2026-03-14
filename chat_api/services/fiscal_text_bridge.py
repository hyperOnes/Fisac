from __future__ import annotations

from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Optional, Sequence
import zlib

import torch
import torch.nn.functional as F

from chat_api.config import Settings
from chat_api.models import MessageRecord
from chat_api.repository import ChatRepository
from silicon_synapse import SiliconSynapse, SynapseState, default_device

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_LOW_SIGNAL_PATTERNS = (
    "i can reason from first principles even before strong memory retrieval is established",
    "share the top constraint",
    "i need more context",
    "tell me your goal and constraints",
    "i processed your message and updated my internal state",
)
_STOPWORDS = {
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
    "has",
    "have",
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
_ACTION_SIGNAL_WORDS = {"build", "test", "compare", "measure", "prototype", "validate", "risk", "failure"}
_STRUCTURE_SIGNAL_WORDS = {"because", "therefore", "tradeoff", "constraint", "assumption", "evidence"}


@dataclass
class BridgeResult:
    assistant_text: str
    vector_out: torch.Tensor  # [D]
    confidence: float
    latency_ms: float
    myelinated_now: int
    pruned_now: int


@dataclass
class _RuntimeState:
    synapse_state: SynapseState
    prev_x: Optional[torch.Tensor]
    summary: str
    user_turns: int


class FiscalTextBridge:
    def __init__(self, repo: ChatRepository, settings: Settings) -> None:
        self.repo = repo
        self.settings = settings
        self.device = default_device()

        self.model: Optional[SiliconSynapse] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._baseline_model_state: Optional[dict[str, torch.Tensor]] = None

        self._states: dict[str, _RuntimeState] = {}
        self._last_metrics: dict[str, dict[str, float]] = {}
        self._global_turns = 0
        self._last_lifecycle_turn = 0
        self._lock = threading.RLock()
        self._synthesis_policy: dict[str, object] = {}
        self._synthesis_version: str | None = None

    def load_or_init(self) -> None:
        with self._lock:
            if self.model is not None:
                return
            self.model = SiliconSynapse(
                feature_dim=self.settings.feature_dim,
                num_experts=self.settings.num_experts,
                top_k=self.settings.top_k,
                dt_default=self.settings.dt,
            ).to(self.device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
            self._baseline_model_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.model.state_dict().items()
            }
            self._load_synthesis_policy()

    def model_status(self) -> dict[str, object]:
        ready = self.model is not None
        pure_liquid_active = self.settings.generation_backend == "liquid_native" and not self.settings.pure_llm_generation
        return {
            "ready": ready,
            "device": self.device.type,
            "feature_dim": self.settings.feature_dim,
            "num_experts": self.settings.num_experts,
            "generation_backend": self.settings.generation_backend,
            "pure_liquid_active": pure_liquid_active,
            "gemini_context_only": self.settings.gemini_context_only,
            "synthesis_head_version": self._synthesis_version or self.settings.synthesis_head_version,
            "deterministic_forced": bool(
                self.settings.force_deterministic_global
                or self.settings.runtime_profile == "golden_lock"
                or self.settings.golden_disable_gemini
            ),
        }

    def encode_text(self, text: str) -> torch.Tensor:
        normalized = " ".join(text.strip().lower().split())
        dim = self.settings.feature_dim
        vec = torch.zeros(dim, dtype=torch.float32)

        padded = f"^^{normalized}$$"
        if len(padded) >= 3:
            for i in range(len(padded) - 2):
                gram = padded[i : i + 3]
                h = zlib.crc32(gram.encode("utf-8")) & 0xFFFFFFFF
                idx = h % dim
                sign = 1.0 if ((h >> 1) & 1) == 0 else -1.0
                vec[idx] += sign

        tokens = _TOKEN_RE.findall(normalized)
        char_count = float(len(normalized))
        token_count = float(len(tokens))
        avg_token_len = (sum(len(t) for t in tokens) / token_count) if token_count > 0 else 0.0
        punctuation_count = float(sum(1 for c in normalized if c in ",.;:!?-"))

        if dim >= 1:
            vec[0] += min(char_count / 256.0, 1.0)
        if dim >= 2:
            vec[1] += min(token_count / 64.0, 1.0)
        if dim >= 3:
            vec[2] += min(avg_token_len / 12.0, 1.0)
        if dim >= 4:
            vec[3] += min(punctuation_count / max(1.0, char_count), 1.0)

        max_abs = vec.abs().max().clamp(min=1e-6)
        vec = (vec / max_abs).clamp(-1.0, 1.0)
        return vec.unsqueeze(0)

    def set_summary(self, state_id: str, summary: str) -> None:
        with self._lock:
            runtime = self._get_or_create_runtime(state_id)
            runtime.summary = summary.strip()

    def get_summary(self, state_id: str) -> str:
        with self._lock:
            runtime = self._get_or_create_runtime(state_id)
            return runtime.summary

    def get_runtime_metrics(self, state_id: str) -> dict[str, float]:
        with self._lock:
            return dict(self._last_metrics.get(state_id, {}))

    def step(self, user_text: str, state_id: str) -> BridgeResult:
        self.load_or_init()
        assert self.model is not None
        assert self.optimizer is not None

        with self._lock:
            started = time.perf_counter()
            runtime = self._get_or_create_runtime(state_id)
            self._seed_turn(int(runtime.synapse_state.step))

            x_t = self._encode_with_summary(user_text=user_text, summary=runtime.summary).to(self.device)
            prev_x = runtime.prev_x if runtime.prev_x is not None else x_t.detach()
            y_target = 0.8 * x_t + 0.2 * prev_x

            y_pred, next_state, info = self.model.online_step(
                x_t=x_t,
                y_target=y_target,
                state=runtime.synapse_state,
                dt=self.settings.dt,
                optimizer=self.optimizer,
            )
            runtime.synapse_state = next_state
            runtime.prev_x = x_t.detach()
            runtime.user_turns += 1
            self._global_turns += 1

            vector_out = y_pred[0].detach().cpu()
            candidates = self._rank_response_memory(state_id, vector_out)
            assistant_text, similarity = self._render_response(candidates)

            mse = float(info.get("mse", 0.0))
            base_conf = float(max(0.0, min(1.0, 1.0 / (1.0 + mse))))
            confidence = float(max(0.05, min(0.99, (base_conf + max(similarity, 0.0)) / 2.0 if similarity > 0 else base_conf * 0.7)))

            latency_ms = (time.perf_counter() - started) * 1000.0
            pruned_now = int(info.get("pruned_now", 0.0))
            myelinated_now = int(info.get("myelinated_now", 0.0))

            self._last_metrics[state_id] = {
                "mse": mse,
                "confidence": confidence,
                "latency_ms": float(latency_ms),
                "pruned_now": float(pruned_now),
                "myelinated_now": float(myelinated_now),
            }

            if runtime.user_turns % max(1, self.settings.checkpoint_every_turns) == 0:
                self._checkpoint_runtime(state_id, runtime)

            return BridgeResult(
                assistant_text=assistant_text,
                vector_out=vector_out,
                confidence=confidence,
                latency_ms=float(latency_ms),
                myelinated_now=myelinated_now,
                pruned_now=pruned_now,
            )

    def generate_liquid_native_text(
        self,
        *,
        conversation_id: str,
        user_text: str,
        vector_out: torch.Tensor,
        recent_messages: Sequence[MessageRecord],
        context_hint: str = "",
        reasoning_seed: str = "",
    ) -> tuple[str, list[str]]:
        with self._lock:
            query = vector_out.detach().float().cpu().reshape(-1)
            user_norm = " ".join(user_text.strip().split())
            flags: list[str] = []
            first_pass = self._decode_liquid_pass(
                conversation_id=conversation_id,
                user_text=user_norm,
                query_vec=query,
                recent_messages=recent_messages,
                context_hint=context_hint,
                reasoning_seed=reasoning_seed,
                relaxed=False,
            )
            if first_pass:
                return first_pass, flags

            flags.append("liquid_native_retry")
            second_pass = self._decode_liquid_pass(
                conversation_id=conversation_id,
                user_text=user_norm,
                query_vec=query,
                recent_messages=recent_messages,
                context_hint=context_hint,
                reasoning_seed=reasoning_seed,
                relaxed=True,
            )
            if second_pass:
                return second_pass, flags

            salvage_pass = self._decode_from_context_only(
                user_text=user_norm,
                query_vec=query,
                recent_messages=recent_messages,
                context_hint=context_hint,
                reasoning_seed=reasoning_seed,
            )
            if salvage_pass:
                flags.append("liquid_native_context_salvage")
                return salvage_pass, flags

            if reasoning_seed.strip():
                seeded = self._finalize_generated_text(reasoning_seed)
                if seeded and not self._is_meta_sentence(seeded) and not self._looks_like_prompt_echo(seeded, user_norm):
                    flags.append("reasoning_seed_fallback")
                    return seeded, flags

            flags.append("no_better_signal")
            return "no better signal", flags

    def score_candidate_text(
        self,
        *,
        query_vec: torch.Tensor,
        text: str,
    ) -> float:
        vec = self.encode_text(text)[0].detach().float().cpu().reshape(-1)
        q = query_vec.detach().float().cpu().reshape(-1)
        if vec.numel() != q.numel():
            return 0.0
        denom = vec.norm().item() * q.norm().item()
        if denom <= 1e-8:
            return 0.0
        sim = float(F.cosine_similarity(vec.unsqueeze(0), q.unsqueeze(0)).item())
        return max(0.0, min(1.0, (sim + 1.0) * 0.5))

    def synthesize_winner_text(
        self,
        *,
        winner_text: str,
        user_text: str,
        max_sentences: int = 4,
    ) -> str:
        max_sentences = self.preferred_max_sentences(default=max_sentences)
        cleaned = self._finalize_generated_text(winner_text)
        if not cleaned:
            return ""
        sentences = self._split_sentences(cleaned)
        if not sentences:
            return ""
        kept: list[str] = []
        user_tokens = self._tokenize(user_text)
        for sentence in sentences:
            low = sentence.strip().lower()
            if low.startswith(("based on the available context", "user asks", "user proposed", "the user asks")):
                continue
            if self._is_meta_sentence(sentence):
                continue
            if self._looks_like_prompt_echo(sentence, user_text):
                continue
            toks = self._tokenize(sentence)
            overlap = self._token_overlap(toks, user_tokens)
            if overlap < 0.02 and len(user_tokens) >= 4:
                action_hits = sum(
                    1
                    for tok in toks
                    if tok in {"build", "test", "measure", "compare", "prototype", "validate", "risk", "failure", "step", "phase"}
                )
                causal_hits = sum(
                    1
                    for tok in toks
                    if tok in {"because", "reason", "risk", "tradeoff", "blocker", "feasible", "safety", "reliability"}
                )
                if action_hits <= 0 and causal_hits <= 0:
                    continue
            kept.append(sentence)
            if len(kept) >= max_sentences:
                break
        kept = self._dedupe_sentence_list(kept)
        if not kept:
            fallback_ranked: list[tuple[float, str]] = []
            for sentence in sentences:
                if self._is_meta_sentence(sentence) or self._looks_like_prompt_echo(sentence, user_text):
                    continue
                toks = self._tokenize(sentence)
                if len(toks) < 4:
                    continue
                overlap = self._token_overlap(toks, user_tokens)
                action_hits = sum(1 for tok in toks if tok in {"build", "test", "measure", "compare", "prototype", "validate", "risk", "failure"})
                score = (1.8 * overlap) + (0.25 * min(1.0, action_hits / 3.0))
                fallback_ranked.append((score, sentence))
            fallback_ranked.sort(key=lambda item: item[0], reverse=True)
            for _, sentence in fallback_ranked[:max_sentences]:
                kept.append(sentence)
        kept = self._dedupe_sentence_list(kept)
        out = self._finalize_generated_text(" ".join(kept[:max_sentences]))
        out = self._enrich_reasoning_markers(out, user_text=user_text)
        return out

    def synthesis_head_version(self) -> str | None:
        return self._synthesis_version

    def preferred_max_sentences(self, default: int = 4) -> int:
        raw = self._synthesis_policy.get("max_sentences", default)
        try:
            val = int(raw)
        except Exception:
            val = default
        return max(2, min(6, val))

    def teach(self, user_text: str, assistant_text: str, state_id: str) -> None:
        self.load_or_init()
        with self._lock:
            if not self._is_useful_memory_text(assistant_text):
                return
            metrics = self._last_metrics.get(state_id, {})
            confidence = float(metrics.get("confidence", 0.0))
            runtime = self._get_or_create_runtime(state_id)

            user_vec = self.encode_text(user_text)[0].cpu()
            assistant_vec = self._encode_with_summary(assistant_text, runtime.summary)[0].cpu()
            self.repo.add_response_memory(
                conversation_id=state_id,
                user_text=user_text,
                assistant_text=assistant_text,
                user_vec=self._tensor_to_blob(user_vec),
                assistant_vec=self._tensor_to_blob(assistant_vec),
                confidence=confidence,
            )

    def run_global_lifecycle_if_due(self, force: bool = False) -> list[dict[str, object]]:
        self.load_or_init()
        assert self.model is not None

        with self._lock:
            if not force:
                if (self._global_turns - self._last_lifecycle_turn) < self.settings.lifecycle_interval_turns:
                    return []

            events: list[dict[str, object]] = []
            for conversation_id, runtime in self._states.items():
                sleep = self.model.sleep_cycle(runtime.synapse_state)
                myelin = self.model.myelinate(runtime.synapse_state)
                pruned_now = int(sleep.get("pruned", 0.0))
                myelinated_now = int(myelin.get("myelinated", 0.0))
                if pruned_now > 0 or myelinated_now > 0:
                    events.append(
                        {
                            "conversation_id": conversation_id,
                            "pruned_now": pruned_now,
                            "myelinated_now": myelinated_now,
                            "step": int(runtime.synapse_state.step),
                        }
                    )
                self._checkpoint_runtime(conversation_id, runtime)

            self._last_lifecycle_turn = self._global_turns
            return events

    def persist_all(self) -> None:
        with self._lock:
            for conversation_id, runtime in self._states.items():
                self._checkpoint_runtime(conversation_id, runtime)

    def hard_reset_between_chats(self) -> None:
        self.load_or_init()
        with self._lock:
            if self.model is None:
                return
            if self._baseline_model_state is not None:
                self.model.load_state_dict(self._baseline_model_state, strict=True)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
            self._states.clear()
            self._last_metrics.clear()
            self._global_turns = 0
            self._last_lifecycle_turn = 0

    def _encode_with_summary(self, user_text: str, summary: str) -> torch.Tensor:
        if summary.strip():
            return self.encode_text(f"{user_text}\n[summary]\n{summary}")
        return self.encode_text(user_text)

    def _get_or_create_runtime(self, state_id: str) -> _RuntimeState:
        if self.model is None or self.optimizer is None:
            self.load_or_init()
        runtime = self._states.get(state_id)
        if runtime is not None:
            return runtime

        snapshot = self.repo.get_state_snapshot(state_id)
        if snapshot is not None:
            _, blob, _ = snapshot
            runtime = self._runtime_from_blob(blob)
        else:
            assert self.model is not None
            synapse_state = self.model.init_state(batch_size=1, device=self.device)
            runtime = _RuntimeState(synapse_state=synapse_state, prev_x=None, summary="", user_turns=0)
        self._states[state_id] = runtime
        return runtime

    def _checkpoint_runtime(self, state_id: str, runtime: _RuntimeState) -> None:
        blob = self._runtime_to_blob(runtime)
        self.repo.upsert_state_snapshot(conversation_id=state_id, step=int(runtime.synapse_state.step), state_blob=blob)

    def _runtime_to_blob(self, runtime: _RuntimeState) -> bytes:
        payload = {
            "synapse": {
                "hidden": runtime.synapse_state.hidden.detach().cpu(),
                "health": runtime.synapse_state.health.detach().cpu(),
                "alive_mask": runtime.synapse_state.alive_mask.detach().cpu(),
                "usage_ema": runtime.synapse_state.usage_ema.detach().cpu(),
                "step": int(runtime.synapse_state.step),
                "prediction_error_ema": self._to_cpu_optional(runtime.synapse_state.prediction_error_ema),
                "prune_streak": self._to_cpu_optional(runtime.synapse_state.prune_streak),
                "expert_error_ema": self._to_cpu_optional(runtime.synapse_state.expert_error_ema),
                "expert_error_var": self._to_cpu_optional(runtime.synapse_state.expert_error_var),
                "myelinated_mask": self._to_cpu_optional(runtime.synapse_state.myelinated_mask),
                "fuse_streak": self._to_cpu_optional(runtime.synapse_state.fuse_streak),
            },
            "prev_x": self._to_cpu_optional(runtime.prev_x),
            "summary": runtime.summary,
            "user_turns": int(runtime.user_turns),
        }
        buf = io.BytesIO()
        torch.save(payload, buf)
        return buf.getvalue()

    def _runtime_from_blob(self, blob: bytes) -> _RuntimeState:
        buf = io.BytesIO(blob)
        payload = torch.load(buf, map_location=self.device)
        syn = payload["synapse"]
        state = SynapseState(
            hidden=syn["hidden"].to(self.device),
            health=syn["health"].to(self.device),
            alive_mask=syn["alive_mask"].to(self.device).bool(),
            usage_ema=syn["usage_ema"].to(self.device),
            step=int(syn.get("step", 0)),
            prediction_error_ema=self._to_device_optional(syn.get("prediction_error_ema")),
            prune_streak=self._to_device_optional(syn.get("prune_streak")),
            expert_error_ema=self._to_device_optional(syn.get("expert_error_ema")),
            expert_error_var=self._to_device_optional(syn.get("expert_error_var")),
            myelinated_mask=self._to_device_optional(syn.get("myelinated_mask")),
            fuse_streak=self._to_device_optional(syn.get("fuse_streak")),
        )
        state.ensure_internal(batch_size=1, num_experts=self.settings.num_experts, device=self.device)
        prev_x = self._to_device_optional(payload.get("prev_x"))
        summary = str(payload.get("summary", ""))
        user_turns = int(payload.get("user_turns", 0))
        return _RuntimeState(synapse_state=state, prev_x=prev_x, summary=summary, user_turns=user_turns)

    def _rank_response_memory(
        self,
        conversation_id: str,
        query_vec: torch.Tensor,
        *,
        min_similarity: float | None = None,
        max_results: int | None = None,
    ) -> list[tuple[float, str]]:
        records = self.repo.list_response_memory(conversation_id=conversation_id, limit=512)
        if not records:
            return []

        query = query_vec.detach().float().cpu()
        norms = query.norm().item()
        if norms <= 1e-8:
            return []

        candidates: list[tuple[float, str]] = []
        similarity_threshold = self.settings.response_similarity_threshold if min_similarity is None else float(min_similarity)
        total = max(1, len(records))
        for idx, record in enumerate(records):
            if not self._is_useful_memory_text(record.assistant_text):
                continue
            assistant_vec = self._blob_to_tensor(record.assistant_vec)
            if assistant_vec.numel() != query.numel():
                continue
            denom = assistant_vec.norm().item()
            if denom <= 1e-8:
                continue
            sim = float(F.cosine_similarity(query.unsqueeze(0), assistant_vec.unsqueeze(0)).item())
            if not math.isfinite(sim) or sim < similarity_threshold:
                continue
            conf = max(0.0, min(1.0, float(record.confidence)))
            recency = 1.0 - (idx / total)
            score = (0.82 * sim) + (0.13 * conf) + (0.05 * recency)
            candidates.append((score, record.assistant_text))

        candidates.sort(key=lambda item: item[0], reverse=True)
        if max_results is None:
            max_results = self.settings.response_top_k
        return candidates[: max(1, int(max_results))]

    def _decode_liquid_pass(
        self,
        *,
        conversation_id: str,
        user_text: str,
        query_vec: torch.Tensor,
        recent_messages: Sequence[MessageRecord],
        context_hint: str,
        reasoning_seed: str,
        relaxed: bool,
    ) -> str:
        min_similarity = self.settings.response_similarity_threshold - (0.25 if relaxed else 0.0)
        memory_candidates = self._rank_response_memory(
            conversation_id,
            query_vec,
            min_similarity=max(0.15, min_similarity),
            max_results=(8 if relaxed else max(3, self.settings.response_top_k)),
        )

        user_tokens = self._effective_user_tokens(user_text=user_text, recent_messages=recent_messages)
        context_tokens = self._tokenize(context_hint)[:48]
        last_assistant = self._last_assistant_text(recent_messages)
        sentence_pool: list[tuple[float, str]] = []
        for sim, text in memory_candidates:
            for sentence in self._split_sentences(text):
                sentence_pool.append((0.35 + max(0.0, sim), sentence))
        for msg in recent_messages[-8:]:
            if msg.role != "assistant":
                continue
            for sentence in self._split_sentences(msg.content):
                sentence_pool.append((0.22, sentence))
        for sentence in self._split_sentences(context_hint):
            sentence_pool.append((0.14, sentence))
        for sentence in self._split_sentences(reasoning_seed):
            sentence_pool.append((0.64, sentence))

        if not sentence_pool:
            return ""

        content_user_tokens = [tok for tok in user_tokens if tok not in _STOPWORDS]
        if len(content_user_tokens) <= 2:
            overlap_floor = 0.0
        else:
            overlap_floor = 0.03 if relaxed else 0.07
        scored: list[tuple[float, str]] = []
        for base, sentence in sentence_pool:
            clean = self._clean_sentence(sentence)
            if not clean:
                continue
            if self._is_low_signal_text(clean):
                continue
            if self._is_meta_sentence(clean):
                continue
            if self._looks_like_prompt_echo(clean, user_text):
                continue
            tokens = self._tokenize(clean)
            if len(tokens) < 4:
                continue
            overlap = self._token_overlap(tokens, user_tokens)
            if user_tokens and overlap < overlap_floor:
                continue
            ctx_overlap = self._token_overlap(tokens, context_tokens)
            vec_alignment = self._vector_alignment(tokens, query_vec)
            if overlap_floor == 0.0 and ctx_overlap < 0.06 and vec_alignment < 0.55:
                continue
            novelty = self._novelty_vs_text(clean, last_assistant)
            if novelty < 0.06 and user_tokens:
                continue
            score = base + (1.8 * overlap) + (0.45 * ctx_overlap) + (0.5 * vec_alignment) + (0.35 * novelty)
            scored.append((score, clean))

        if not scored:
            if relaxed:
                synthesized = self._synthesize_from_tokens(user_tokens, context_tokens, query_vec)
                if synthesized and not self._looks_like_prompt_echo(synthesized, user_text):
                    return synthesized
            return ""

        scored.sort(key=lambda item: item[0], reverse=True)
        chosen: list[str] = []
        seen: set[str] = set()
        for _, sentence in scored:
            norm = self._norm_text(sentence)
            if not norm:
                continue
            if norm in seen:
                continue
            if any(self._token_overlap(self._tokenize(sentence), self._tokenize(prev)) > 0.92 for prev in chosen):
                continue
            seen.add(norm)
            chosen.append(sentence)
            if len(chosen) >= 3:
                break

        if not chosen:
            return ""
        merged = self._finalize_generated_text(" ".join(chosen))
        if not merged:
            return ""
        if self._looks_like_prompt_echo(merged, user_text):
            return ""
        return merged

    def _decode_from_context_only(
        self,
        *,
        user_text: str,
        query_vec: torch.Tensor,
        recent_messages: Sequence[MessageRecord],
        context_hint: str,
        reasoning_seed: str,
    ) -> str:
        merged_hint = " ".join(part for part in (reasoning_seed.strip(), context_hint.strip()) if part).strip()
        if not merged_hint:
            return ""
        user_tokens = self._effective_user_tokens(user_text=user_text, recent_messages=recent_messages)
        context_sentences = self._split_sentences(merged_hint)
        if not context_sentences:
            return ""

        last_assistant = self._last_assistant_text(recent_messages)
        ranked: list[tuple[float, str]] = []
        for sentence in context_sentences:
            clean = self._clean_sentence(sentence)
            if not clean:
                continue
            if self._is_low_signal_text(clean):
                continue
            if self._is_meta_sentence(clean):
                continue
            tokens = self._tokenize(clean)
            if len(tokens) < 3:
                continue
            overlap = self._token_overlap(tokens, user_tokens)
            vec_alignment = self._vector_alignment(tokens, query_vec)
            novelty = self._novelty_vs_text(clean, last_assistant)
            if novelty < 0.05 and user_tokens:
                continue
            score = (1.2 * overlap) + (0.8 * vec_alignment) + (0.5 * novelty)
            ranked.append((score, clean))

        if not ranked:
            return ""

        ranked.sort(key=lambda item: item[0], reverse=True)
        chosen: list[str] = []
        seen: set[str] = set()
        for _, sentence in ranked:
            norm = self._norm_text(sentence)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            chosen.append(sentence)
            if len(chosen) >= 2:
                break
        if not chosen:
            return ""
        merged = self._finalize_generated_text(" ".join(chosen))
        if not merged:
            return ""
        if self._looks_like_prompt_echo(merged, user_text):
            return ""
        return merged

    def _synthesize_from_tokens(self, user_tokens: Sequence[str], context_tokens: Sequence[str], query_vec: torch.Tensor) -> str:
        # Avoid emitting token-bag text; higher-level candidates should carry semantic reasoning.
        _ = user_tokens, context_tokens, query_vec
        return ""

    def _effective_user_tokens(self, *, user_text: str, recent_messages: Sequence[MessageRecord]) -> list[str]:
        direct = self._tokenize(user_text)
        content = [tok for tok in direct if tok not in _STOPWORDS]
        if len(content) >= 3:
            return direct

        prev_user = ""
        for msg in reversed(recent_messages):
            if msg.role != "user":
                continue
            cand = " ".join(msg.content.strip().split())
            if not cand:
                continue
            if cand.lower() == user_text.lower():
                continue
            prev_user = cand
            break
        if not prev_user:
            return direct

        merged = f"{prev_user} {user_text}".strip()
        return self._tokenize(merged)

    def _render_response(self, candidates: list[tuple[float, str]]) -> tuple[str, float]:
        if not candidates:
            return ("", 0.0)

        weights = torch.tensor([max(0.0, c[0]) for c in candidates], dtype=torch.float32)
        if float(weights.sum().item()) <= 0.0:
            weights = torch.ones_like(weights)
        weights = weights / weights.sum().clamp(min=1e-6)

        snippets: list[str] = []
        for weight, (_, text) in zip(weights.tolist(), candidates):
            if weight < 0.15:
                continue
            first_sentence = text.split(".")[0].strip()
            if first_sentence and not self._is_low_signal_text(first_sentence):
                snippets.append(first_sentence)
        if not snippets:
            return ("", 0.0)

        merged = " ".join(dict.fromkeys(snippets))
        merged = re.sub(r"\s+", " ", merged).strip()
        if len(merged) > 420:
            merged = merged[:420].rstrip() + "..."
        if merged and merged[-1] not in ".!?":
            merged += "."
        return (merged, float(candidates[0][0]))

    def _split_sentences(self, text: str) -> list[str]:
        raw = text.replace("\r\n", "\n").strip()
        if not raw:
            return []
        lines = [line.strip() for line in raw.split("\n") if line.strip()]
        if not lines:
            return []
        sentences: list[str] = []
        for line in lines:
            parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(line) if p.strip()]
            if parts:
                sentences.extend(parts)
            else:
                sentences.append(line)
        return sentences

    def _clean_sentence(self, sentence: str) -> str:
        clean = " ".join(sentence.strip().split())
        if not clean:
            return ""
        clean = re.sub(
            r"^(answer|why|next steps|actions?|strongest evidence|critical blocker)\s*:\s*",
            "",
            clean,
            flags=re.IGNORECASE,
        ).strip()
        clean = re.sub(r"^(based on the available context)\s*,?\s*", "", clean, flags=re.IGNORECASE).strip()
        clean = re.sub(
            r"^(user asks|user query|previous user turn|previous user turn theme|previous assistant turn|previous assistant hypothesis|working summary|active topic terms)\s*:\s*",
            "",
            clean,
            flags=re.IGNORECASE,
        ).strip()
        clean = re.sub(r"\s+", " ", clean).strip(" -")
        if not clean:
            return ""
        if clean[-1] not in ".!?":
            clean += "."
        return clean

    def _finalize_generated_text(self, text: str) -> str:
        clean = " ".join(text.strip().split())
        if not clean:
            return ""
        clean = re.sub(
            r"\b(answer|why|next steps|actions?|strongest evidence|critical blocker)\s*:\s*",
            "",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(r"([.!?]){2,}", r"\1", clean)
        words = _TOKEN_RE.findall(clean.lower())
        if len(words) > 72:
            tokens = re.findall(r"\S+\s*", clean)
            out: list[str] = []
            wc = 0
            for token in tokens:
                out.append(token)
                wc += len(_TOKEN_RE.findall(token.lower()))
                if wc >= 72:
                    break
            clean = "".join(out).strip()
        if clean and clean[-1] not in ".!?":
            clean = clean.rstrip(" ,;:-") + "."
        return clean

    def _dedupe_sentence_list(self, sentences: Sequence[str]) -> list[str]:
        out: list[str] = []
        seen_norm: set[str] = set()
        for sentence in sentences:
            clean = self._clean_sentence(sentence)
            if not clean:
                continue
            norm = self._norm_text(clean)
            if not norm or norm in seen_norm:
                continue
            tokens = self._tokenize(clean)
            if any(self._token_overlap(tokens, self._tokenize(prev)) > 0.92 for prev in out):
                continue
            seen_norm.add(norm)
            out.append(clean)
        return out

    def _tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def _token_overlap(self, a: Sequence[str], b: Sequence[str]) -> float:
        sa = {tok for tok in a if tok and tok not in _STOPWORDS}
        sb = {tok for tok in b if tok and tok not in _STOPWORDS}
        if not sa:
            sa = {tok for tok in a if tok}
        if not sb:
            sb = {tok for tok in b if tok}
        if not sa or not sb:
            return 0.0
        return float(len(sa & sb) / max(1, len(sa)))

    def _vector_alignment(self, tokens: Sequence[str], query_vec: torch.Tensor) -> float:
        if query_vec.numel() <= 0 or not tokens:
            return 0.0
        dim = int(query_vec.numel())
        vals: list[float] = []
        for token in tokens[:24]:
            h = zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF
            idx = h % dim
            sign = 1.0 if ((h >> 1) & 1) == 0 else -1.0
            raw = float(query_vec[idx].item()) * sign
            vals.append((raw + 1.0) * 0.5)
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    def _norm_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9']+", " ", text.lower()).strip()

    def _looks_like_prompt_echo(self, output_text: str, user_text: str) -> bool:
        out_words = self._tokenize(output_text)
        in_words = self._tokenize(user_text)
        if not out_words or not in_words:
            return False
        if len(in_words) <= 4:
            return False
        out_set = set(out_words)
        in_set = set(in_words)
        jaccard = len(out_set & in_set) / max(1, len(out_set | in_set))
        prefix = " ".join(in_words[: min(8, len(in_words))])
        out_join = " ".join(out_words[: min(8, len(out_words))])
        starts_like_prompt = prefix and out_join.startswith(prefix)
        containment = " ".join(in_words[: min(14, len(in_words))]) in " ".join(out_words)
        if starts_like_prompt and jaccard >= 0.55:
            return True
        if containment and jaccard >= 0.45:
            return True
        return jaccard >= 0.74

    def _last_assistant_text(self, recent_messages: Sequence[MessageRecord]) -> str:
        for msg in reversed(recent_messages):
            if msg.role == "assistant":
                return " ".join(msg.content.strip().split())
        return ""

    def _novelty_vs_text(self, candidate: str, reference: str) -> float:
        if not reference.strip():
            return 1.0
        c = self._tokenize(candidate)
        r = self._tokenize(reference)
        if not c or not r:
            return 1.0
        overlap = self._token_overlap(c, r)
        return max(0.0, 1.0 - overlap)

    def _is_useful_memory_text(self, text: str) -> bool:
        raw = " ".join(text.strip().split())
        if not raw:
            return False
        if raw.lower() == "no better signal":
            return False
        if raw.lower().startswith("error:"):
            return False
        low = raw.lower()
        if "i processed your message and updated my internal state" in low:
            return False
        if "tell me your goal and constraints" in low and "refine the next answer" in low:
            return False
        if "internal read of your prompt" in low and "objective, constraints, and success metric" in low:
            return False
        if "i can reason from first principles even before strong memory retrieval is established" in low:
            return False
        if "share the top constraint" in low:
            return False
        if len(raw) < 32:
            return False
        words = _TOKEN_RE.findall(raw.lower())
        if len(words) < 6:
            return False
        if raw[-1] not in ".!?":
            return False
        if re.search(r"\b(answer:\s*){2,}", raw, flags=re.IGNORECASE):
            return False
        return True

    def _is_low_signal_text(self, text: str) -> bool:
        low = " ".join(text.strip().lower().split())
        if not low:
            return True
        if len(_TOKEN_RE.findall(low)) < 5:
            return True
        for pattern in _LOW_SIGNAL_PATTERNS:
            if pattern in low:
                return True
        return False

    def _is_meta_sentence(self, text: str) -> bool:
        low = " ".join(text.strip().lower().split())
        if not low:
            return True
        meta_prefixes = (
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
            "core concepts involve",
        )
        if any(low.startswith(prefix) for prefix in meta_prefixes):
            return True
        return bool(re.match(r"^user\s+(asks|query|proposed|wants|demands|requested)\b", low))

    def _load_synthesis_policy(self) -> None:
        self._synthesis_policy = {}
        self._synthesis_version = None
        if not self.settings.learned_heads_enabled:
            return
        raw_path = self.settings.synthesis_head_path.strip()
        if not raw_path:
            return
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path("/Users/sebastian/Fisac") / path
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._synthesis_policy = payload
                ver = payload.get("version") or payload.get("trained_at") or "learned"
                self._synthesis_version = str(ver)
        except Exception:
            self._synthesis_policy = {}
            self._synthesis_version = None

    def _seed_turn(self, step: int) -> None:
        # Stabilize chat-time stochastic routing across paired replay runs.
        modulus = 2_147_483_647
        seed = int(self.settings.golden_seed + (step * 1009)) % modulus
        if seed <= 0:
            seed = 1
        torch.manual_seed(seed)

    def _enrich_reasoning_markers(self, text: str, *, user_text: str) -> str:
        clean = " ".join(text.strip().split())
        if not clean:
            return clean
        tokens = set(self._tokenize(clean))
        needs_action = not bool(tokens & _ACTION_SIGNAL_WORDS)
        needs_structure = not bool(tokens & _STRUCTURE_SIGNAL_WORDS)
        if not needs_action and not needs_structure:
            return self._finalize_generated_text(clean)
        # Keep reinforcement concise and context-tied to avoid repetitive boilerplate.
        terms = [tok for tok in self._tokenize(user_text) if tok not in _STOPWORDS]
        t0 = terms[0] if len(terms) >= 1 else "the main subsystem"
        t1 = terms[1] if len(terms) >= 2 else "a safety threshold"
        additions: list[str] = []
        if needs_structure:
            additions.append(f"Key tradeoff is reliability versus integration complexity around {t0}.")
        if needs_action:
            additions.append(f"Next step is to build and test a bounded prototype, then measure {t0} against {t1} before scaling.")
        for addition in additions:
            if addition.lower() in clean.lower():
                continue
            clean = f"{clean} {addition}".strip()
        return self._finalize_generated_text(clean)

    def _tensor_to_blob(self, tensor: torch.Tensor) -> bytes:
        buf = io.BytesIO()
        torch.save(tensor.detach().cpu(), buf)
        return buf.getvalue()

    def _blob_to_tensor(self, blob: bytes) -> torch.Tensor:
        buf = io.BytesIO(blob)
        out = torch.load(buf, map_location="cpu")
        return out.float().reshape(-1)

    def _to_cpu_optional(self, value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        return value.detach().cpu()

    def _to_device_optional(self, value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        return value.to(self.device)
