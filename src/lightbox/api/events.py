"""Lightweight in-process SSE EventBus for `job.progress`, `plan.updated`, `agent.event`, `memory.proposed` (SPEC §9.2)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ServerEvent:
    event_type: str
    data: dict[str, Any]
    session_id: str | None = None
    ts: float = 0.0

    def format_sse(self) -> str:
        payload = {"ts": self.ts, "session_id": self.session_id, **self.data}
        return f"event: {self.event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class EventBus:
    """Thread-safe async pub/sub hub backing `GET /api/events` SSE streams."""

    def __init__(self, *, max_queue_size: int = 256) -> None:
        self._lock = threading.Lock()
        self._max_queue_size = max_queue_size
        self._closed = False
        self._subscribers: list[
            tuple[asyncio.AbstractEventLoop, asyncio.Queue[ServerEvent | None], str | None]
        ] = []

    def close(self) -> None:
        """Signal all active SSE subscribers to terminate immediately (e.g. on SIGINT/shutdown)."""
        with self._lock:
            self._closed = True
            subs = list(self._subscribers)

        for loop, queue, _filter_sid in subs:
            if loop.is_closed():
                continue
            try:
                loop.call_soon_threadsafe(self._enqueue_nowait, queue, None)
            except RuntimeError:
                pass

    def publish(
        self,
        event_type: str,
        data: dict[str, Any],
        *,
        session_id: str | None = None,
    ) -> None:
        """Publish an event from any thread (sync worker or asyncio loop)."""
        event = ServerEvent(
            event_type=event_type,
            data=data,
            session_id=session_id,
            ts=time.time(),
        )
        with self._lock:
            if self._closed:
                return
            subs = list(self._subscribers)

        for loop, queue, filter_sid in subs:
            if filter_sid is not None and session_id is not None and filter_sid != session_id:
                continue
            if loop.is_closed():
                continue
            try:
                loop.call_soon_threadsafe(self._enqueue_nowait, queue, event)
            except RuntimeError:
                pass

    @staticmethod
    def _enqueue_nowait(
        queue: asyncio.Queue[ServerEvent | None],
        event: ServerEvent | None,
    ) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    async def subscribe(
        self,
        *,
        session_id: str | None = None,
        heartbeat_s: float = 15.0,
    ) -> AsyncIterator[str]:
        """Yield SSE-formatted chunks until client disconnects or `close()` is called."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[ServerEvent | None] = asyncio.Queue(maxsize=self._max_queue_size)
        entry = (loop, queue, session_id)
        with self._lock:
            if self._closed:
                return
            self._subscribers.append(entry)

        try:
            # Emit immediate connected frame so clients confirm subscription without waiting
            yield ServerEvent(
                event_type="connected",
                data={"ok": True},
                session_id=session_id,
                ts=time.time(),
            ).format_sse()
            while not self._closed:
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=heartbeat_s)
                    if ev is None:
                        break
                    yield ev.format_sse()
                except TimeoutError:
                    if self._closed:
                        break
                    yield ": keepalive\n\n"
        finally:
            with self._lock:
                if entry in self._subscribers:
                    self._subscribers.remove(entry)


__all__ = ["EventBus", "ServerEvent"]
