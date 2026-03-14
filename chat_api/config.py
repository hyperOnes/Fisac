from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Literal

from chat_api.system_prompt import DEFAULT_FISAC_SYSTEM_PROMPT


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _parse_csv(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    out = []
    for part in raw.split(","):
        val = part.strip()
        if val:
            out.append(val)
    return tuple(out)


def _load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _merged_env() -> dict[str, str]:
    env = dict(os.environ)
    fallback_files = (
        Path("/Users/sebastian/Fisac/chat_api/.env.local"),
        Path("/Users/sebastian/Fisac/.env.local"),
        Path("/Users/sebastian/Fisac/.env"),
    )
    for file_path in fallback_files:
        for key, value in _load_env_file(file_path).items():
            env.setdefault(key, value)
    return env


def _resolve_secret(env: dict[str, str], key: str, file_key: str) -> str:
    direct = env.get(key, "").strip()
    if direct:
        return direct
    file_path = env.get(file_key, "").strip()
    if not file_path:
        return ""
    path = Path(file_path)
    if not path.is_absolute():
        path = Path("/Users/sebastian/Fisac") / path
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000
    db_path: Path = Path("/Users/sebastian/Fisac/chat_api/fiscal_chat.db")
    feature_dim: int = 64
    num_experts: int = 128
    top_k: int = 4
    dt: float = 0.02
    checkpoint_every_turns: int = 10
    context_window_messages: int = 20
    summary_every_user_turns: int = 6
    summary_max_sentences: int = 3
    response_top_k: int = 3
    response_similarity_threshold: float = 0.55
    response_guard_enabled: bool = True
    response_depth_mode: Literal["concise", "balanced", "detailed"] = "balanced"
    response_shape: Literal["freeform", "hybrid", "structured"] = "hybrid"
    response_min_words_balanced: int = 70
    response_min_sentences_balanced: int = 3
    response_max_words_balanced: int = 260
    response_regen_attempts: int = 1
    response_incomplete_tail_guard: bool = True
    runtime_profile: Literal["default", "golden_lock"] = "default"
    force_deterministic_global: bool = False
    golden_disable_gemini: bool = False
    per_chat_hard_reset: bool = True
    golden_baseline_checkpoint: Path = Path("/Users/sebastian/Fisac/artifacts/golden_recovery/best/golden_baseline.pt")
    golden_seed: int = 9521
    golden_confidence_floor: float = 0.86
    golden_confidence_ceiling: float = 0.92
    memory_scope: Literal["conversation", "global"] = "conversation"
    max_active_conversation_models: int = 8
    pure_llm_generation: bool = False
    generation_backend: Literal["liquid_native", "gemini_raw", "hybrid"] = "liquid_native"
    jury_mode: bool = True
    jury_head_version: str = "v0"
    synthesis_head_version: str = "v0"
    learned_heads_enabled: bool = False
    jury_head_path: str = ""
    synthesis_head_path: str = ""
    jury_max_candidates: int = 5
    lifecycle_interval_turns: int = 1000
    gemini_default_enabled: bool = True
    gemini_api_key: str = ""
    gemini_api_keys: tuple[str, ...] = ()
    gemini_api_key_file: str = ""
    gemini_model: str = "gemini-3.1-pro-preview"
    gemini_endpoint: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_timeout_seconds: float = 8.0
    gemini_retries: int = 2
    gemini_temperature: float = 0.45
    gemini_top_p: float = 0.92
    gemini_max_output_tokens: int = 700
    gemini_thinking_budget: int = -1
    gemini_context_only: bool = True
    openai_api_key: str = ""
    openai_api_keys: tuple[str, ...] = ()
    openai_api_key_file: str = ""
    openai_model: str = "gpt-4.1-mini"
    openai_endpoint: str = "https://api.openai.com/v1"
    openai_timeout_seconds: float = 8.0
    openai_retries: int = 1
    external_context_enabled: bool = True
    context_query_count: int = 5
    web_context_enabled: bool = True
    web_context_timeout_seconds: float = 2.5
    web_context_max_items: int = 3
    system_prompt: str = DEFAULT_FISAC_SYSTEM_PROMPT
    system_prompt_enabled: bool = True
    cursor_sequence_length: int = 30
    cursor_predict_horizon_start: int = 15
    cursor_predict_points: int = 5
    cursor_infer_every_n_frames: int = 3
    cursor_dev_controls: bool = False
    cursor_weights_dir: Path = Path("/Users/sebastian/Fisac/organic_cursor/weights")
    cursor_hidden_dim: int = 64
    cursor_gru_weights: str = "gru_cursor.pt"
    cursor_liquid_weights: str = "liquid_cursor.pt"


def get_settings() -> Settings:
    env = _merged_env()
    gemini_primary = _resolve_secret(env, "FISCAL_GEMINI_API_KEY", "FISCAL_GEMINI_API_KEY_FILE")
    gemini_many = list(_parse_csv(env.get("FISCAL_GEMINI_API_KEYS")))
    if gemini_primary:
        gemini_many.insert(0, gemini_primary)
    # Keep order while de-duplicating.
    gemini_keys = tuple(dict.fromkeys(gemini_many))
    openai_key = _resolve_secret(env, "FISCAL_OPENAI_API_KEY", "FISCAL_OPENAI_API_KEY_FILE")
    openai_many = list(_parse_csv(env.get("FISCAL_OPENAI_API_KEYS")))
    if openai_key:
        openai_many.insert(0, openai_key)
    openai_keys = tuple(dict.fromkeys(openai_many))
    return Settings(
        host=env.get("FISCAL_CHAT_HOST", "127.0.0.1"),
        port=int(env.get("FISCAL_CHAT_PORT", "8000")),
        db_path=Path(env.get("FISCAL_CHAT_DB", "/Users/sebastian/Fisac/chat_api/fiscal_chat.db")),
        feature_dim=int(env.get("FISCAL_CHAT_FEATURE_DIM", "64")),
        num_experts=int(env.get("FISCAL_CHAT_NUM_EXPERTS", "128")),
        top_k=int(env.get("FISCAL_CHAT_TOP_K", "4")),
        dt=float(env.get("FISCAL_CHAT_DT", "0.02")),
        checkpoint_every_turns=int(env.get("FISCAL_CHAT_CHECKPOINT_TURNS", "10")),
        context_window_messages=int(env.get("FISCAL_CHAT_CONTEXT_WINDOW", "20")),
        summary_every_user_turns=int(env.get("FISCAL_CHAT_SUMMARY_EVERY", "6")),
        summary_max_sentences=int(env.get("FISCAL_CHAT_SUMMARY_SENTENCES", "3")),
        response_top_k=int(env.get("FISCAL_CHAT_RESPONSE_TOP_K", "3")),
        response_similarity_threshold=float(env.get("FISCAL_CHAT_SIM_THRESHOLD", "0.55")),
        response_guard_enabled=_parse_bool(env.get("FISCAL_CHAT_RESPONSE_GUARD_ENABLED"), True),
        response_depth_mode=env.get("FISCAL_CHAT_RESPONSE_DEPTH_MODE", "balanced"),  # type: ignore[arg-type]
        response_shape=env.get("FISCAL_CHAT_RESPONSE_SHAPE", "hybrid"),  # type: ignore[arg-type]
        response_min_words_balanced=int(env.get("FISCAL_CHAT_RESPONSE_MIN_WORDS_BALANCED", "70")),
        response_min_sentences_balanced=int(env.get("FISCAL_CHAT_RESPONSE_MIN_SENTENCES_BALANCED", "3")),
        response_max_words_balanced=int(env.get("FISCAL_CHAT_RESPONSE_MAX_WORDS_BALANCED", "260")),
        response_regen_attempts=int(env.get("FISCAL_CHAT_RESPONSE_REGEN_ATTEMPTS", "1")),
        response_incomplete_tail_guard=_parse_bool(env.get("FISCAL_CHAT_RESPONSE_INCOMPLETE_TAIL_GUARD"), True),
        runtime_profile=env.get("FISCAL_CHAT_RUNTIME_PROFILE", "default"),  # type: ignore[arg-type]
        force_deterministic_global=_parse_bool(env.get("FISCAL_CHAT_FORCE_DETERMINISTIC_GLOBAL"), False),
        golden_disable_gemini=_parse_bool(env.get("FISCAL_CHAT_GOLDEN_DISABLE_GEMINI"), False),
        per_chat_hard_reset=_parse_bool(env.get("FISCAL_CHAT_PER_CHAT_HARD_RESET"), True),
        golden_baseline_checkpoint=Path(
            env.get(
                "FISCAL_CHAT_GOLDEN_BASELINE_CHECKPOINT",
                "/Users/sebastian/Fisac/artifacts/golden_recovery/best/golden_baseline.pt",
            )
        ),
        golden_seed=int(env.get("FISCAL_CHAT_GOLDEN_SEED", "9521")),
        golden_confidence_floor=float(env.get("FISCAL_CHAT_GOLDEN_CONFIDENCE_FLOOR", "0.86")),
        golden_confidence_ceiling=float(env.get("FISCAL_CHAT_GOLDEN_CONFIDENCE_CEILING", "0.92")),
        memory_scope=env.get("FISCAL_CHAT_MEMORY_SCOPE", "conversation"),  # type: ignore[arg-type]
        max_active_conversation_models=int(env.get("FISCAL_CHAT_MAX_ACTIVE_CONVERSATION_MODELS", "8")),
        pure_llm_generation=_parse_bool(env.get("FISCAL_CHAT_PURE_LLM_GENERATION"), False),
        generation_backend=env.get("FISCAL_CHAT_GENERATION_BACKEND", "liquid_native"),  # type: ignore[arg-type]
        jury_mode=_parse_bool(env.get("FISCAL_CHAT_JURY_MODE"), True),
        jury_head_version=env.get("FISCAL_CHAT_JURY_HEAD_VERSION", "v0"),
        synthesis_head_version=env.get("FISCAL_CHAT_SYNTHESIS_HEAD_VERSION", "v0"),
        learned_heads_enabled=_parse_bool(env.get("FISCAL_CHAT_LEARNED_HEADS_ENABLED"), False),
        jury_head_path=env.get("FISCAL_CHAT_JURY_HEAD_PATH", ""),
        synthesis_head_path=env.get("FISCAL_CHAT_SYNTHESIS_HEAD_PATH", ""),
        jury_max_candidates=int(env.get("FISCAL_CHAT_JURY_MAX_CANDIDATES", "5")),
        lifecycle_interval_turns=int(env.get("FISCAL_CHAT_LIFECYCLE_TURNS", "1000")),
        gemini_default_enabled=_parse_bool(env.get("FISCAL_GEMINI_DEFAULT_ENABLED"), True),
        gemini_api_key=gemini_primary,
        gemini_api_keys=gemini_keys,
        gemini_api_key_file=env.get("FISCAL_GEMINI_API_KEY_FILE", ""),
        gemini_model=env.get("FISCAL_GEMINI_MODEL", "gemini-3.1-pro-preview"),
        gemini_endpoint=env.get("FISCAL_GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta"),
        gemini_timeout_seconds=float(env.get("FISCAL_GEMINI_TIMEOUT_SECONDS", "8.0")),
        gemini_retries=int(env.get("FISCAL_GEMINI_RETRIES", "2")),
        gemini_temperature=float(env.get("FISCAL_GEMINI_TEMPERATURE", "0.45")),
        gemini_top_p=float(env.get("FISCAL_GEMINI_TOP_P", "0.92")),
        gemini_max_output_tokens=int(env.get("FISCAL_GEMINI_MAX_OUTPUT_TOKENS", "700")),
        gemini_thinking_budget=int(env.get("FISCAL_GEMINI_THINKING_BUDGET", "-1")),
        gemini_context_only=_parse_bool(env.get("FISCAL_GEMINI_CONTEXT_ONLY"), True),
        openai_api_key=openai_key,
        openai_api_keys=openai_keys,
        openai_api_key_file=env.get("FISCAL_OPENAI_API_KEY_FILE", ""),
        openai_model=env.get("FISCAL_OPENAI_MODEL", "gpt-4.1-mini"),
        openai_endpoint=env.get("FISCAL_OPENAI_ENDPOINT", "https://api.openai.com/v1"),
        openai_timeout_seconds=float(env.get("FISCAL_OPENAI_TIMEOUT_SECONDS", "8.0")),
        openai_retries=int(env.get("FISCAL_OPENAI_RETRIES", "1")),
        external_context_enabled=_parse_bool(env.get("FISCAL_EXTERNAL_CONTEXT_ENABLED"), True),
        context_query_count=int(env.get("FISCAL_CONTEXT_QUERY_COUNT", "5")),
        web_context_enabled=_parse_bool(env.get("FISCAL_WEB_CONTEXT_ENABLED"), True),
        web_context_timeout_seconds=float(env.get("FISCAL_WEB_CONTEXT_TIMEOUT_SECONDS", "2.5")),
        web_context_max_items=int(env.get("FISCAL_WEB_CONTEXT_MAX_ITEMS", "3")),
        system_prompt=env.get("FISCAL_CHAT_SYSTEM_PROMPT", DEFAULT_FISAC_SYSTEM_PROMPT),
        system_prompt_enabled=_parse_bool(env.get("FISCAL_CHAT_SYSTEM_PROMPT_ENABLED"), True),
        cursor_sequence_length=int(env.get("FISCAL_CURSOR_SEQUENCE_LENGTH", "30")),
        cursor_predict_horizon_start=int(env.get("FISCAL_CURSOR_HORIZON_START", "15")),
        cursor_predict_points=int(env.get("FISCAL_CURSOR_PREDICT_POINTS", "5")),
        cursor_infer_every_n_frames=int(env.get("FISCAL_CURSOR_INFER_EVERY_N", "3")),
        cursor_dev_controls=_parse_bool(env.get("FISCAL_CURSOR_DEV_CONTROLS"), False),
        cursor_weights_dir=Path(env.get("FISCAL_CURSOR_WEIGHTS_DIR", "/Users/sebastian/Fisac/organic_cursor/weights")),
        cursor_hidden_dim=int(env.get("FISCAL_CURSOR_HIDDEN_DIM", "64")),
        cursor_gru_weights=env.get("FISCAL_CURSOR_GRU_WEIGHTS", "gru_cursor.pt"),
        cursor_liquid_weights=env.get("FISCAL_CURSOR_LIQUID_WEIGHTS", "liquid_cursor.pt"),
    )
