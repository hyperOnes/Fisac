#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

import httpx

CANONICAL_PROMPTS = [
    "What is your current take? What you need from me? hi btw :)",
    "Explain",
    "Is it shit idea",
    "What do you advice instead?",
]

BANNED_PATTERNS = [
    re.compile(r"\b(?:answer|why|next\s*steps?)\s*:", re.IGNORECASE),
]
BANNED_PHRASES = [
    "i processed your message and updated my internal state",
    "i can reason from first principles even before strong memory retrieval is established",
    "tell me your goal and constraints",
]
TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass
class TurnEval:
    prompt: str
    output: str
    latency_ms: float
    length_ok: bool
    label_ok: bool
    boilerplate_ok: bool
    overlap: float
    overlap_ok: bool


@dataclass
class CandidateReport:
    candidate: str
    ok: bool
    error: str | None
    turns: list[TurnEval]
    latency_p95_turn2_4: float
    checks_passed: int
    checks_total: int
    score: float


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _parse_sse(raw: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for frame in raw.split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        event_name: str | None = None
        event_data: dict[str, Any] | None = None
        for line in frame.splitlines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                payload = line.split(":", 1)[1].strip()
                try:
                    event_data = json.loads(payload)
                except Exception:
                    event_data = {"raw": payload}
        if event_name and event_data is not None:
            events.append((event_name, event_data))
    return events


def _token_overlap(prompt: str, output: str) -> float:
    a = {tok for tok in TOKEN_RE.findall(prompt.lower()) if len(tok) >= 3}
    b = {tok for tok in TOKEN_RE.findall(output.lower()) if len(tok) >= 3}
    if not a:
        return 0.0
    return len(a & b) / float(len(a))


def _prepare_workspace(base_repo: Path, candidate_bundle: Path, *, overlay_code: bool) -> Path:
    tmp_root = Path(tempfile.mkdtemp(prefix="fisac_golden_eval_"))
    work = tmp_root / "workspace"
    work.mkdir(parents=True, exist_ok=True)

    # Copy minimal runtime tree.
    shutil.copytree(base_repo / "chat_api", work / "chat_api", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(base_repo / "organic_cursor", work / "organic_cursor", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "weights"))
    shutil.copy2(base_repo / "silicon_synapse.py", work / "silicon_synapse.py")

    # Overlay candidate files.
    for src in candidate_bundle.rglob("*"):
        if not src.is_file() or src.name == "manifest.json":
            continue
        rel = src.relative_to(candidate_bundle)
        # Optional compatibility mode: keep current code, only overlay DB/state.
        if not overlay_code and str(rel).startswith("chat_api/services"):
            continue
        if not overlay_code and rel == Path("chat_api/config.py"):
            continue
        dst = work / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    return work


def _wait_for_health(base_url: str, timeout_s: float = 20.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/api/health", timeout=1.5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def _evaluate_turn(prompt: str, output: str, latency_ms: float) -> TurnEval:
    output_strip = output.strip()
    length_ok = 40 <= len(output_strip) <= 260
    label_ok = not any(p.search(output_strip) for p in BANNED_PATTERNS)
    low = output_strip.lower()
    boilerplate_ok = all(phrase not in low for phrase in BANNED_PHRASES)
    overlap = _token_overlap(prompt, output_strip)
    overlap_ok = overlap >= 0.15
    return TurnEval(
        prompt=prompt,
        output=output_strip,
        latency_ms=latency_ms,
        length_ok=length_ok,
        label_ok=label_ok,
        boilerplate_ok=boilerplate_ok,
        overlap=overlap,
        overlap_ok=overlap_ok,
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = max(0, min(len(vals) - 1, int(round((pct / 100.0) * (len(vals) - 1)))))
    return float(vals[idx])


def _evaluate_workspace(candidate_name: str, work: Path) -> CandidateReport:
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    env = dict(os.environ)
    env["FISCAL_CHAT_DB"] = str(work / "chat_api" / "fiscal_chat.db")
    env["FISCAL_CHAT_RUNTIME_PROFILE"] = "golden_lock"
    env["FISCAL_CHAT_FORCE_DETERMINISTIC_GLOBAL"] = "1"
    env["FISCAL_CHAT_GOLDEN_DISABLE_GEMINI"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "chat_api.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(work),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        if not _wait_for_health(base_url):
            output = ""
            try:
                proc.terminate()
                output, _ = proc.communicate(timeout=3)
            except Exception:
                try:
                    proc.kill()
                    output, _ = proc.communicate(timeout=2)
                except Exception:
                    output = ""
            return CandidateReport(
                candidate=candidate_name,
                ok=False,
                error=f"backend_not_ready: {output[-1000:]}",
                turns=[],
                latency_p95_turn2_4=0.0,
                checks_passed=0,
                checks_total=0,
                score=0.0,
            )

        with httpx.Client(base_url=base_url, timeout=20.0) as client:
            conv = client.post("/api/conversations", json={"title": f"golden-eval-{candidate_name}"})
            conv.raise_for_status()
            conversation_id = conv.json()["id"]

            turn_reports: list[TurnEval] = []
            for prompt in CANONICAL_PROMPTS:
                resp = client.post(
                    "/api/chat/respond",
                    json={"conversation_id": conversation_id, "message": prompt, "stream": True},
                )
                resp.raise_for_status()
                events = _parse_sse(resp.text)
                tokens: list[str] = []
                done_latency = 0.0
                for event_name, data in events:
                    if event_name == "token":
                        delta = data.get("delta")
                        if isinstance(delta, str):
                            tokens.append(delta)
                    if event_name == "done":
                        done_latency = float(data.get("latency_ms", 0.0) or 0.0)
                output = "".join(tokens).strip()
                turn_reports.append(_evaluate_turn(prompt=prompt, output=output, latency_ms=done_latency))

        # Score turns 2-4.
        scored = turn_reports[1:4]
        checks_total = len(scored) * 4 + 1  # + latency criterion
        checks_passed = 0
        for t in scored:
            checks_passed += int(t.length_ok)
            checks_passed += int(t.label_ok)
            checks_passed += int(t.boilerplate_ok)
            checks_passed += int(t.overlap_ok)
        p95 = _percentile([t.latency_ms for t in scored], 95)
        latency_ok = p95 <= 250.0
        checks_passed += int(latency_ok)

        score = checks_passed + max(0.0, 1.0 - (p95 / 250.0))
        return CandidateReport(
            candidate=candidate_name,
            ok=True,
            error=None,
            turns=turn_reports,
            latency_p95_turn2_4=p95,
            checks_passed=checks_passed,
            checks_total=checks_total,
            score=score,
        )
    except Exception as exc:
        output = ""
        try:
            proc.terminate()
            output, _ = proc.communicate(timeout=3)
        except Exception:
            try:
                proc.kill()
                output, _ = proc.communicate(timeout=2)
            except Exception:
                output = ""
        return CandidateReport(
            candidate=candidate_name,
            ok=False,
            error=f"{exc}\n{output}".strip(),
            turns=[],
            latency_p95_turn2_4=0.0,
            checks_passed=0,
            checks_total=0,
            score=0.0,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


def evaluate_candidate(base_repo: Path, candidate_bundle: Path) -> CandidateReport:
    # First pass: candidate code + state.
    work = _prepare_workspace(base_repo=base_repo, candidate_bundle=candidate_bundle, overlay_code=True)
    primary = _evaluate_workspace(candidate_name=candidate_bundle.name, work=work)
    if primary.ok:
        return primary

    # Fallback pass: current code + candidate DB/state files only.
    fallback_work = _prepare_workspace(base_repo=base_repo, candidate_bundle=candidate_bundle, overlay_code=False)
    secondary = _evaluate_workspace(candidate_name=candidate_bundle.name, work=fallback_work)
    if secondary.ok:
        secondary.error = (primary.error or "")[:800] + "\nused_current_code_fallback=true"
    return secondary if secondary.ok else primary


def _find_candidates(candidates_dir: Path) -> list[Path]:
    if not candidates_dir.exists():
        return []
    out: list[Path] = []
    for p in sorted(candidates_dir.iterdir()):
        if not p.is_dir():
            continue
        if p.name in {"best", "reports"}:
            continue
        if (p / "manifest.json").exists():
            out.append(p)
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay canonical golden-window prompt script across candidate bundles and rank style-quality match.")
    parser.add_argument("--candidates-dir", type=Path, default=Path("artifacts/golden_recovery"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/golden_recovery/reports"))
    args = parser.parse_args()

    base_repo = Path(__file__).resolve().parents[1]
    candidates = _find_candidates(args.candidates_dir)
    if not candidates:
        print("No candidates found.")
        return 1

    reports: list[CandidateReport] = []
    for candidate in candidates:
        report = evaluate_candidate(base_repo=base_repo, candidate_bundle=candidate)
        reports.append(report)
        _write_json(args.output_dir / f"{candidate.name}_report.json", asdict(report))

    ranked = sorted(
        reports,
        key=lambda r: (int(r.ok), r.checks_passed, r.score),
        reverse=True,
    )

    leaderboard = {
        "generated_at": _now_iso(),
        "candidates": [asdict(r) for r in ranked],
    }
    _write_json(args.output_dir / "leaderboard.json", leaderboard)

    best = ranked[0]
    best_dir = args.candidates_dir / "best"
    source_dir = args.candidates_dir / best.candidate
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(source_dir, best_dir)
    best_manifest = {
        "selected_candidate": best.candidate,
        "selected_at": _now_iso(),
        "score": best.score,
        "checks_passed": best.checks_passed,
        "checks_total": best.checks_total,
        "latency_p95_turn2_4": best.latency_p95_turn2_4,
        "ok": best.ok,
    }
    _write_json(args.candidates_dir / "golden_baseline_manifest.json", best_manifest)

    print(json.dumps(leaderboard, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
