from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import threading
import time


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class CaptureConfig:
    output: Path
    minutes: float = 15.0


class MouseRecorder:
    def __init__(self, cfg: CaptureConfig) -> None:
        self.cfg = cfg
        self._rows: list[tuple[int, float, float, str]] = []
        self._stop = threading.Event()

    def _on_move(self, x: float, y: float) -> None:
        if self._stop.is_set():
            return
        self._rows.append((_now_ms(), float(x), float(y), "move"))

    def _on_click(self, x: float, y: float, _button, pressed: bool) -> None:
        if self._stop.is_set():
            return
        if pressed:
            self._rows.append((_now_ms(), float(x), float(y), "click"))

    def _on_key_press(self, key) -> bool | None:
        # ESC triggers graceful early stop.
        if str(key) == "Key.esc":
            self._stop.set()
            return False
        return None

    def run(self) -> None:
        try:
            from pynput import keyboard, mouse
        except Exception as exc:
            raise RuntimeError("pynput is required for recording mouse data. Install with `pip install pynput`.") from exc

        self.cfg.output.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + max(1.0, self.cfg.minutes * 60.0)

        mouse_listener = mouse.Listener(on_move=self._on_move, on_click=self._on_click)
        key_listener = keyboard.Listener(on_press=self._on_key_press)
        mouse_listener.start()
        key_listener.start()
        print(f"Recording mouse stream to {self.cfg.output} for up to {self.cfg.minutes:.1f} minutes.")
        print("Press ESC or Ctrl+C to stop early.")
        try:
            while not self._stop.is_set():
                if time.time() >= deadline:
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            self._stop.set()
        finally:
            self._stop.set()
            mouse_listener.stop()
            key_listener.stop()
            mouse_listener.join(timeout=1.0)
            key_listener.join(timeout=1.0)
            self._write_csv()

    def _write_csv(self) -> None:
        with self.cfg.output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp_ms", "x", "y", "event"])
            writer.writerows(self._rows)
        print(f"Wrote {len(self._rows)} rows to {self.cfg.output}")


def parse_args() -> CaptureConfig:
    parser = argparse.ArgumentParser(description="Capture mouse cursor stream for organic cursor training.")
    parser.add_argument("--output", type=Path, default=Path("organic_cursor/data/mouse_capture.csv"))
    parser.add_argument("--minutes", type=float, default=15.0)
    args = parser.parse_args()
    return CaptureConfig(output=args.output, minutes=args.minutes)


def main() -> None:
    cfg = parse_args()
    MouseRecorder(cfg).run()


if __name__ == "__main__":
    main()
