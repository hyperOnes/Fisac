from __future__ import annotations

import asyncio
from typing import Optional

from chat_api.services.chat_service import ChatService


class LifecycleWorker:
    def __init__(self, chat_service: ChatService, interval_seconds: float = 2.0) -> None:
        self.chat_service = chat_service
        self.interval_seconds = max(0.5, interval_seconds)
        self._task: Optional[asyncio.Task[None]] = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.chat_service.run_lifecycle_maintenance()
            except Exception:
                # Keep worker alive; errors are recorded by request path metrics.
                pass
            await asyncio.sleep(self.interval_seconds)
