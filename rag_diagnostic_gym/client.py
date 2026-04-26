"""
RAGDiagnosticClient — async WebSocket client.

Mirrors the OpenEnv EnvClient interface:
  async with RAGDiagnosticClient(base_url=...) as client:
      obs = await client.reset()
      obs = await client.step(DiagnoseAction(...))
      obs = await client.step(PatchAction(...))

Install: pip install websockets
"""
from __future__ import annotations

import json
import asyncio
from typing import Any

import websockets

from rag_diagnostic_gym.models import DiagnoseAction, Observation, PatchAction


class RAGDiagnosticClient:
    def __init__(self, base_url: str = "ws://localhost:8000/ws", timeout: float = 30.0):
        self._url     = base_url
        self._timeout = timeout
        self._ws: Any = None

    async def __aenter__(self) -> "RAGDiagnosticClient":
        self._ws = await websockets.connect(self._url, open_timeout=self._timeout)
        return self

    async def __aexit__(self, *_) -> None:
        if self._ws:
            await self._ws.close()

    async def reset(self, task_id: str | None = None) -> Observation:
        data: dict = {"task_id": task_id} if task_id else {}
        await self._ws.send(json.dumps({"type": "reset", "data": data}))
        return await self._recv()

    async def step(self, action: DiagnoseAction | PatchAction) -> Observation:
        await self._ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
        return await self._recv()

    async def _recv(self) -> Observation:
        raw = await asyncio.wait_for(self._ws.recv(), timeout=self._timeout)
        msg = json.loads(raw)
        if msg["type"] == "error":
            raise RuntimeError(f"Server error: {msg['data']['message']}")
        payload = msg.get("data", {})
        # OpenEnv server variants may nest observation under `data.observation`.
        if isinstance(payload, dict) and "observation" in payload and isinstance(payload["observation"], dict):
            obs_payload = dict(payload["observation"])
            # Preserve top-level reward if provided separately.
            if "reward" in payload and "reward" not in obs_payload:
                obs_payload["reward"] = payload["reward"]
            payload = obs_payload
        return Observation(**payload)
