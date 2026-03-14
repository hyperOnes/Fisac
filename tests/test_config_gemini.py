from __future__ import annotations

from pathlib import Path

from chat_api.config import get_settings


def test_default_gemini_model_is_31_preview() -> None:
    settings = get_settings()
    assert settings.gemini_model == "gemini-3.1-pro-preview"


def test_api_key_file_resolution(monkeypatch, tmp_path) -> None:
    secret_path = Path(tmp_path) / "gemini.key"
    secret_path.write_text("key-from-file", encoding="utf-8")
    monkeypatch.setenv("FISCAL_GEMINI_API_KEY", "")
    monkeypatch.setenv("FISCAL_GEMINI_API_KEY_FILE", str(secret_path))
    settings = get_settings()
    assert settings.gemini_api_key == "key-from-file"
