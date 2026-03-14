#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

REQUIRED_FILES = [
    Path("chat_api/services/chat_service.py"),
    Path("chat_api/services/fiscal_text_bridge.py"),
    Path("chat_api/services/conversational_composer.py"),
    Path("chat_api/config.py"),
]
OPTIONAL_FILES = [
    Path("chat_api/fiscal_chat.db"),
    Path("chat_api/fiscal_chat.db-wal"),
    Path("chat_api/fiscal_chat.db-shm"),
]

IGNORE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".venv", "node_modules"}
IGNORE_SUFFIXES = {".pyc", ".pyo"}


@dataclass
class CandidateBundle:
    source_id: str
    source_kind: str
    path: Path
    copied_files: list[Path]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_local_snapshots() -> list[str]:
    try:
        proc = subprocess.run(
            ["tmutil", "listlocalsnapshots", "/"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []

    out: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "com.apple.TimeMachine." in line:
            out.append(line.split("/", 1)[-1])
    return sorted(set(out))


def _snapshot_dt(snapshot_id: str) -> datetime | None:
    # com.apple.TimeMachine.2026-02-25-221425.local
    try:
        stem = snapshot_id.split("com.apple.TimeMachine.", 1)[1].split(".local", 1)[0]
        return datetime.strptime(stem, "%Y-%m-%d-%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def choose_bracketing_snapshots(snapshot_ids: Iterable[str], target: datetime) -> list[str]:
    parsed: list[tuple[str, datetime]] = []
    for sid in snapshot_ids:
        dt = _snapshot_dt(sid)
        if dt is not None:
            parsed.append((sid, dt))
    if not parsed:
        return []

    parsed.sort(key=lambda x: x[1])
    before: tuple[str, datetime] | None = None
    after: tuple[str, datetime] | None = None
    for item in parsed:
        if item[1] <= target:
            before = item
        if item[1] >= target and after is None:
            after = item

    picks: list[str] = []
    if before:
        picks.append(before[0])
    if after and (before is None or after[0] != before[0]):
        picks.append(after[0])
    if not picks:
        picks.append(parsed[0][0])
    return picks


def _discover_snapshot_roots(snapshot_id: str) -> list[Path]:
    candidate_roots: list[Path] = []

    # Known direct patterns.
    known_prefixes = [
        Path("/.MobileBackups"),
        Path("/Volumes/com.apple.TimeMachine.localsnapshots/Backups.backupdb"),
        Path("/System/Volumes/Data/.MobileBackups"),
    ]
    for prefix in known_prefixes:
        if not prefix.exists():
            continue
        if prefix.name == "Backups.backupdb":
            # Layout often: <prefix>/<host>/<snapshot>/Macintosh HD - Data/Users/sebastian/Fisac
            try:
                host_dirs = list(prefix.iterdir())
            except PermissionError:
                host_dirs = []
            for host_dir in host_dirs:
                snap_dir = host_dir / snapshot_id
                if snap_dir.exists():
                    candidate_roots.extend(
                        [
                            snap_dir / "Macintosh HD - Data" / "Users" / "sebastian" / "Fisac",
                            snap_dir / "Macintosh HD" / "Users" / "sebastian" / "Fisac",
                            snap_dir / "Users" / "sebastian" / "Fisac",
                        ]
                    )
        else:
            snap_dir = prefix / snapshot_id
            candidate_roots.extend(
                [
                    snap_dir / "Macintosh HD - Data" / "Users" / "sebastian" / "Fisac",
                    snap_dir / "Macintosh HD" / "Users" / "sebastian" / "Fisac",
                    snap_dir / "Users" / "sebastian" / "Fisac",
                ]
            )

    return [p for p in candidate_roots if p.exists()]


def _copy_bundle_from_root(source_root: Path, out_dir: Path, source_id: str, source_kind: str) -> CandidateBundle:
    bundle_dir = out_dir / source_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    candidate_rel_paths: set[Path] = set(REQUIRED_FILES + OPTIONAL_FILES)
    # Copy a self-consistent chat backend module set to avoid mixed-version runtime errors.
    chat_api_root = source_root / "chat_api"
    if chat_api_root.exists():
        for p in chat_api_root.rglob("*.py"):
            if any(part in IGNORE_DIR_NAMES for part in p.parts):
                continue
            candidate_rel_paths.add(p.relative_to(source_root))

    for rel in (Path("silicon_synapse.py"), Path("organic_cursor/app.py")):
        if (source_root / rel).exists():
            candidate_rel_paths.add(rel)

    copied_files: list[Path] = []
    for rel in sorted(candidate_rel_paths):
        src = source_root / rel
        if not src.exists():
            if rel in REQUIRED_FILES:
                continue
            continue
        if src.is_dir():
            continue
        if any(part in IGNORE_DIR_NAMES for part in src.parts):
            continue
        if src.suffix in IGNORE_SUFFIXES:
            continue
        dst = bundle_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_files.append(rel)

    manifest_path = bundle_dir / "manifest.json"
    manifest = {
        "source_id": source_id,
        "source_kind": source_kind,
        "captured_at": _now_iso(),
        "source_root": str(source_root),
        "files": [],
    }
    for rel in copied_files:
        p = bundle_dir / rel
        manifest["files"].append(
            {
                "path": str(rel),
                "size": p.stat().st_size,
                "mtime": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
                "sha256": _sha256(p),
            }
        )

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return CandidateBundle(source_id=source_id, source_kind=source_kind, path=bundle_dir, copied_files=copied_files)


def recover_candidates(target_iso: str, output_dir: Path, fallback_source: Path) -> list[CandidateBundle]:
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        target = datetime.fromisoformat(target_iso)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SystemExit(f"Invalid --target-iso: {exc}") from exc

    bundles: list[CandidateBundle] = []

    snapshot_ids = list_local_snapshots()
    picks = choose_bracketing_snapshots(snapshot_ids, target.astimezone(timezone.utc))
    for snapshot_id in picks:
        roots = _discover_snapshot_roots(snapshot_id)
        if not roots:
            continue
        # Prefer first valid root for deterministic behavior.
        bundle = _copy_bundle_from_root(
            source_root=roots[0],
            out_dir=output_dir,
            source_id=snapshot_id,
            source_kind="timemachine",
        )
        bundles.append(bundle)

    if not bundles:
        if not fallback_source.exists():
            return []
        bundle = _copy_bundle_from_root(
            source_root=fallback_source,
            out_dir=output_dir,
            source_id="fallback_backup",
            source_kind="fallback_backup",
        )
        bundles.append(bundle)

    return bundles


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover golden-window candidate bundles from Time Machine snapshots or fallback backup.")
    parser.add_argument("--target-iso", default="2026-02-25T21:51:21+01:00")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/golden_recovery"))
    parser.add_argument("--fallback-source", type=Path, default=Path("/Users/sebastian/Backups/fisax feb 26/Fisac"))
    args = parser.parse_args()

    bundles = recover_candidates(
        target_iso=args.target_iso,
        output_dir=args.output_dir,
        fallback_source=args.fallback_source,
    )
    if not bundles:
        print("No candidates recovered.")
        return 1

    summary = {
        "target_iso": args.target_iso,
        "generated_at": _now_iso(),
        "candidates": [
            {
                "source_id": b.source_id,
                "source_kind": b.source_kind,
                "path": str(b.path),
                "copied_files": [str(p) for p in b.copied_files],
            }
            for b in bundles
        ],
    }
    (args.output_dir / "recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
