from __future__ import annotations

import importlib
import os
import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient


def _load_app_with_db(db_path: str):
    os.environ["FISCAL_CHAT_DB"] = db_path
    for mod in ["chat_api.main", "chat_api.config"]:
        if mod in sys.modules:
            del sys.modules[mod]
    import chat_api.main as main

    importlib.reload(main)
    return main


def test_cursor_ws_pred_payload(tmp_path) -> None:
    main = _load_app_with_db(str(tmp_path / "cursor_ws.db"))
    with TestClient(main.app) as client:
        with client.websocket_connect("/ws/cursor") as ws:
            ready_seen = False
            last = None
            for i in range(45):
                ws.send_json(
                    {
                        "type": "frame",
                        "sample_index": i,
                        "x": float(100 + i * 2),
                        "y": float(200 + i),
                        "t_ms": 1000 + i * 16,
                        "viewport_w": 1440,
                        "viewport_h": 900,
                    }
                )
                msg = ws.receive_json()
                assert msg["type"] == "pred"
                assert "metrics" in msg
                assert "gru_path" in msg
                assert "liquid_path" in msg
                assert "kalman_path" in msg
                assert "abg_path" in msg
                assert "buffer_ready" in msg
                assert isinstance(msg.get("sample_index"), int)
                if msg["buffer_ready"]:
                    ready_seen = True
                last = msg

            assert ready_seen is True
            assert last is not None
            assert len(last["gru_path"]) == 5
            assert len(last["liquid_path"]) == 5
            assert len(last["kalman_path"]) == 5
            assert len(last["abg_path"]) == 5

            metrics = last["metrics"]
            assert isinstance(metrics.get("gru_mse_ema"), float)
            assert isinstance(metrics.get("liquid_mse_ema"), float)
            assert isinstance(metrics.get("mse_advantage"), float)
            assert isinstance(metrics.get("rtt_ms"), float)

            assert isinstance(metrics.get("fde250_gru_px_ema"), float)
            assert isinstance(metrics.get("fde250_liquid_px_ema"), float)
            assert isinstance(metrics.get("fde250_kalman_px_ema"), float)
            assert isinstance(metrics.get("fde250_abg_px_ema"), float)
            assert isinstance(metrics.get("ade_gru_px_ema"), float)
            assert isinstance(metrics.get("ade_liquid_px_ema"), float)
            assert isinstance(metrics.get("ade_kalman_px_ema"), float)
            assert isinstance(metrics.get("ade_abg_px_ema"), float)
            assert isinstance(metrics.get("fde_final_gru_px_ema"), float)
            assert isinstance(metrics.get("fde_final_liquid_px_ema"), float)
            assert isinstance(metrics.get("fde_final_kalman_px_ema"), float)
            assert isinstance(metrics.get("fde_final_abg_px_ema"), float)

            infer_gru = metrics.get("infer_gru_ms_ema")
            infer_liquid = metrics.get("infer_liquid_ms_ema")
            assert infer_gru is None or isinstance(infer_gru, float)
            assert infer_liquid is None or isinstance(infer_liquid, float)


def test_cursor_ws_survives_invalid_and_non_frame_payloads(tmp_path) -> None:
    main = _load_app_with_db(str(tmp_path / "cursor_ws_resilience.db"))
    with TestClient(main.app) as client:
        with client.websocket_connect("/ws/cursor") as ws:
            ws.send_json({"type": "noop"})
            msg = ws.receive_json()
            assert msg["type"] == "error"

            ws.send_json(
                {
                    "type": "frame",
                    "sample_index": 0,
                    "x": "not-a-number",
                    "y": None,
                    "t_ms": "invalid",
                    "viewport_w": "bad",
                    "viewport_h": -4,
                }
            )
            msg = ws.receive_json()
            assert msg["type"] == "pred"
            assert len(msg["gru_path"]) == 5
            assert len(msg["kalman_path"]) == 5
            assert len(msg["abg_path"]) == 5

            ws.send_json(
                {
                    "type": "frame",
                    "sample_index": 1,
                    "x": 150.0,
                    "y": 260.0,
                    "t_ms": 1040,
                    "viewport_w": 1280,
                    "viewport_h": 720,
                }
            )
            msg = ws.receive_json()
            assert msg["type"] == "pred"
