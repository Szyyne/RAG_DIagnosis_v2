"""
RAG Diagnostic — OpenEnv v0.2.2 server.

Uses the Environment base class pattern from openenv-core:
  • FastAPI app with WebSocket at /ws
  • HTTP endpoints: GET /health, GET /state, GET /tasks
  • Web UI at /web (set ENABLE_WEB_INTERFACE=true)

WebSocket message format
─────────────────────────
  Client → Server:  {"type": "reset"|"step", "data": <action_dict>}
  Server → Client:  {"type": "observation", "data": <observation_dict>}
                 or {"type": "error",       "data": {"message": str}}

Two-step episode
─────────────────
  Step 0: DiagnoseAction   → partial reward ∈ [0, 1]
  Step 1: PatchAction      → full reward ∈ [0, 1], terminated=True

Reward normalization
─────────────────────
  All rewards are in [0, 1].
  Difficulty multipliers (1.0 / 1.3 / 1.6) are applied then divided by the
  maximum multiplier (1.6) so the final episode reward never exceeds 1.0.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from openenv.core.env_server import Environment, create_fastapi_app

from rag_diagnostic_gym.models import (
    DiagnoseAction,
    EnvStepAction,
    Observation,
    PatchAction,
    RewardBreakdown,
)
from rag_diagnostic_gym.reward import (
    MAX_MULTIPLIER,
    CompositeDiagnosticRubric,
    apply_rubrics,
    score_diagnosis,
    score_faithfulness,
    score_patch,
)
from rag_diagnostic_gym.tasks import TASKS, DIFFICULTY_MULTIPLIER


# ─────────────────────────────────────────────────────────────────
# Core environment logic (stateful per-session)
# ─────────────────────────────────────────────────────────────────

class RAGDiagnosticEnvironment(Environment[EnvStepAction | DiagnoseAction | PatchAction, Observation, Observation | None]):
    """
    OpenEnv-compatible environment for RAG pipeline fault diagnosis.

    Episode structure
    -----------------
    reset()  → Observation (step=0, symptoms visible, reward=0)
    step(DiagnoseAction) → Observation (step=1, partial reward ∈ [0,1])
    step(PatchAction)    → Observation (step=2, terminated, full reward ∈ [0,1])

    All scalar rewards are normalised to [0, 1].
    """

    # OpenEnv metadata
    ENV_ID      = "rag-diagnostic-gym-v1"
    MAX_STEPS   = 2
    REWARD_RANGE = (0.0, 1.0)

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, seed: int = 42):
        super().__init__(rubric=CompositeDiagnosticRubric())
        self._rng            = random.Random(seed)
        self._task: Dict[str, Any] = {}
        self._obs: Observation | None = None
        self._step_count     = 0
        self._diag_payload: Dict[str, Any] = {}
        self._episode_reward: float = 0.0

    # ── Public gym-style API ───────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset to a new (or specified) task. Returns initial Observation."""
        if seed is not None:
            self._rng.seed(seed)
        tid = task_id or self._rng.choice(list(TASKS.keys()))
        if tid not in TASKS:
            raise ValueError(f"Unknown task_id '{tid}'. Valid: {list(TASKS.keys())}")
        self._task           = TASKS[tid]
        self._step_count     = 0
        self._diag_payload   = {}
        self._episode_reward = 0.0
        self._reset_rubric()
        self._obs = Observation(
            task_id    = tid,
            difficulty = self._task["difficulty"],
            step       = 0,
            symptoms   = self._task["observable_state"],
            reward     = 0.0,
            done       = False,
            info       = {
                "env_id":       self.ENV_ID,
                "max_steps":    self.MAX_STEPS,
                "reward_range": list(self.REWARD_RANGE),
                "available_tasks": list(TASKS.keys()),
            },
        )
        return self._obs

    def step(
        self,
        action: EnvStepAction | DiagnoseAction | PatchAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Observation:
        """Advance the episode by one action. Returns next Observation."""
        if isinstance(action, EnvStepAction):
            action = action.root
        if self._obs is None:
            raise RuntimeError("Call reset() before step().")
        if self._obs.terminated:
            raise RuntimeError("Episode terminated. Call reset().")

        if isinstance(action, DiagnoseAction):
            return self._step_diagnose(action)
        if isinstance(action, PatchAction):
            return self._step_patch(action)
        raise TypeError(f"Unexpected action type: {type(action)}")

    @property
    def state(self) -> Observation | None:
        """Return the current observation (None before first reset)."""
        return self._obs

    # ── Private step handlers ──────────────────────────────────────

    def _step_diagnose(self, action: DiagnoseAction) -> Observation:
        if self._step_count != 0:
            raise RuntimeError("DiagnoseAction must be the first step (step=0).")

        tid = self._obs.task_id
        d_score, d_info = score_diagnosis(
            action.root_cause,
            action.explanation,
            action.confidence,
            tid,
        )
        mult = DIFFICULTY_MULTIPLIER[self._task["difficulty"]]
        # OpenEnv rubric-computed step reward (already normalized to [0, 1])
        reward = float(min(1.0, max(0.0, self._apply_rubric(action, self._obs))))

        self._episode_reward += reward
        self._diag_payload   = action.model_dump()
        self._step_count     = 1

        self._obs = Observation(
            task_id    = tid,
            difficulty = self._task["difficulty"],
            step       = 1,
            symptoms   = self._task["observable_state"],
            diagnosis  = self._diag_payload,
            reward     = round(reward, 4),        # ∈ [0, 1]
            done       = False,
            reward_breakdown = RewardBreakdown(
                diagnosis_score       = d_info["diagnosis_score"],
                difficulty_multiplier = mult,
                scalar_reward         = round(reward, 4),
            ),
            terminated = False,
            info = {
                "step":                    1,
                "ground_truth_root_cause": self._task["root_cause"],
                "diagnosis_breakdown":     d_info,
                "rubric_scores": {
                    "diagnosis": d_info["diagnosis_score"],
                    "patch":     None,
                    "faithfulness": None,
                },
                "hint": (
                    f"Correct root cause: hidden until episode end. "
                    f"Your diagnosis scored {d_info['diagnosis_score']:.2f}/1.0. "
                    "Now submit a PatchAction to fix the pipeline."
                ),
            },
        )
        return self._obs

    def _step_patch(self, action: PatchAction) -> Observation:
        if self._step_count != 1:
            raise RuntimeError("PatchAction must be the second step (step=1).")

        tid = self._obs.task_id
        p_score, p_info = score_patch(action.patch, tid)
        f_score         = score_faithfulness(action.patch, tid)
        mult            = DIFFICULTY_MULTIPLIER[self._task["difficulty"]]

        # OpenEnv rubric-computed step reward (already normalized to [0, 1])
        patch_reward = float(min(1.0, max(0.0, self._apply_rubric(action, self._obs))))

        self._episode_reward = float(min(1.0, self._episode_reward + patch_reward))
        self._step_count     = 2

        # Full rubric scores for transparency
        rubrics = apply_rubrics(tid, self._diag_payload, action.model_dump())

        self._obs = Observation(
            task_id    = tid,
            difficulty = self._task["difficulty"],
            step       = 2,
            symptoms   = self._task["observable_state"],
            diagnosis  = self._diag_payload,
            reward     = round(patch_reward, 4),             # step reward ∈ [0, 1]
            done       = True,
            reward_breakdown = RewardBreakdown(
                patch_f1             = p_info["patch_f1"],
                faithfulness_delta   = f_score,
                difficulty_multiplier = mult,
                scalar_reward        = round(self._episode_reward, 4),  # ∈ [0, 1]
            ),
            terminated = True,
            info = {
                "step":                   2,
                "ground_truth_root_cause": self._task["root_cause"],
                "ground_truth_patch":      self._task["correct_patch"],
                "submitted_patch":         action.patch,
                "patch_breakdown":         p_info,
                "faithfulness_score":      f_score,
                "episode_total_reward":    round(self._episode_reward, 4),  # ∈ [0, 1]
                "rubric_scores":           rubrics,      # composable rubric breakdown
                "expected_post_patch":     self._task["expected_post_patch"],
            },
        )
        return self._obs


# ─────────────────────────────────────────────────────────────────
# FastAPI app factory (built on OpenEnv create_fastapi_app)
# ─────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    def make_env() -> RAGDiagnosticEnvironment:
        """Factory — each WebSocket connection gets an isolated environment."""
        return RAGDiagnosticEnvironment()

    app = create_fastapi_app(
        env=make_env,
        action_cls=EnvStepAction,
        observation_cls=Observation,
    )
    app.title = "RAG Diagnostic Gym"
    app.description = (
        "OpenEnv v0.2.2 RL environment for RAG pipeline fault diagnosis. "
        "All rewards are normalized to [0, 1]."
    )
    app.version = "0.2.2"

    # ── HTTP convenience endpoints ─────────────────────────────────

    @app.get("/health")
    async def health() -> dict:
        return {
            "status":       "ok",
            "env":          "rag-diagnostic-gym",
            "version":      app.version,
            "reward_range": [0.0, 1.0],
        }

    @app.get("/tasks")
    async def list_tasks() -> dict:
        return {
            tid: {
                "difficulty":         t["difficulty"],
                "root_cause":         t["root_cause"],
                "reward_multiplier":  DIFFICULTY_MULTIPLIER[t["difficulty"]],
                "correct_patch_keys": list(t["correct_patch"].keys()),
            }
            for tid, t in TASKS.items()
        }

    @app.get("/state")
    async def state_endpoint() -> JSONResponse:
        """Returns a snapshot of a demo environment (not session-aware)."""
        env = make_env()
        obs = env.reset()
        return JSONResponse(obs.model_dump())

    @app.get("/rubrics")
    async def rubrics_endpoint() -> dict:
        """Describe the composable rubric system."""
        return {
            "components": {
                "diagnosis":    {"weight": 0.35, "range": [0.0, 1.0], "agent": 1},
                "patch":        {"weight": 0.45, "range": [0.0, 1.0], "agent": 2},
                "faithfulness": {"weight": 0.20, "range": [0.0, 1.0], "agent": 2},
            },
            "difficulty_multipliers": DIFFICULTY_MULTIPLIER,
            "normalization": f"composite / {MAX_MULTIPLIER} → final ∈ [0, 1]",
        }

    # ── Optional web UI ────────────────────────────────────────────

    if os.getenv("ENABLE_WEB_INTERFACE", "false").lower() == "true":
        @app.get("/web", response_class=HTMLResponse)
        async def web_ui() -> str:
            return """<!DOCTYPE html>
<html>
<head><title>RAG Diagnostic Gym</title></head>
<body style="font-family:monospace;padding:2em">
<h2>🔬 RAG Diagnostic Gym — Web Interface</h2>
<p>Connect via WebSocket at <code>ws://localhost:8000/ws</code></p>
<p>See <a href="/tasks">/tasks</a>, <a href="/rubrics">/rubrics</a>,
<a href="/health">/health</a></p>
</body></html>"""

    return app


app = create_app()   # module-level for uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "rag_diagnostic_gym.server.environment:app",
        host    = "0.0.0.0",
        port    = int(os.getenv("PORT", 8000)),
        reload  = os.getenv("DEV", "false").lower() == "true",
        workers = 1,
    )
