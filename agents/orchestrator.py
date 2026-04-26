"""
Two-agent orchestrator for RAG Diagnostic Gym.

Agent 1 — DiagnosticAgent: sees symptoms → returns root_cause + explanation + confidence
Agent 2 — PatchAgent:      sees symptoms + diagnosis → returns config patch

Both agents call an LLM backend (any OpenAI-compatible endpoint or local model).
During GRPO training, the LLM is the model being fine-tuned.
"""
from __future__ import annotations

import asyncio
import json
import re
import textwrap
from typing import Any, Callable

from rag_diagnostic_gym.client import RAGDiagnosticClient
from rag_diagnostic_gym.models import DiagnoseAction, Observation, PatchAction


# ─────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────

def diagnostic_prompt(obs: Observation) -> str:
    return textwrap.dedent(f"""
        You are a senior RAG pipeline engineer on-call.
        The pipeline below is broken. Identify the single root cause.

        ## Observable Symptoms
        ```json
        {json.dumps(obs.symptoms, indent=2)}
        ```

        ## Required output — valid JSON only, no markdown fences, no extra text:
        {{
          "root_cause": "<snake_case_identifier>",
          "explanation": "<2-4 sentences explaining why this root cause produces all symptoms>",
          "confidence": <float 0.0-1.0>
        }}
    """).strip()


def patch_prompt(obs: Observation) -> str:
    return textwrap.dedent(f"""
        You are a senior RAG pipeline engineer on-call.
        Agent 1 has diagnosed the root cause. Fix it with a minimal config patch.

        ## Observable Symptoms
        ```json
        {json.dumps(obs.symptoms, indent=2)}
        ```

        ## Agent 1 Diagnosis
        ```json
        {json.dumps(obs.diagnosis, indent=2)}
        ```

        ## Required output — valid JSON only, no markdown fences, no extra text:
        {{
          "patch": {{
            "<config_key>": <corrected_value>
          }},
          "rationale": "<1-2 sentences>"
        }}
    """).strip()


# ─────────────────────────────────────────────────────────────────
# JSON parser (robust to LLM noise)
# ─────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise


# ─────────────────────────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────────────────────────

ModelFn = Callable[[str], str]   # prompt → completion string


class DiagnosticAgent:
    """Agent 1 — produces DiagnoseAction from raw symptoms."""

    def __init__(self, model_fn: ModelFn):
        self._call = model_fn

    def act(self, obs: Observation) -> DiagnoseAction:
        prompt  = diagnostic_prompt(obs)
        raw     = self._call(prompt)
        payload = _parse_json(raw)
        return DiagnoseAction(
            root_cause  = payload.get("root_cause", "unknown"),
            explanation = payload.get("explanation", ""),
            confidence  = float(payload.get("confidence", 0.5)),
        )

    def get_prompt(self, obs: Observation) -> str:
        return diagnostic_prompt(obs)


class PatchAgent:
    """Agent 2 — produces PatchAction from diagnosis + symptoms."""

    def __init__(self, model_fn: ModelFn):
        self._call = model_fn

    def act(self, obs: Observation) -> PatchAction:
        if obs.diagnosis is None:
            raise ValueError("PatchAgent requires obs.diagnosis (run DiagnosticAgent first).")
        prompt  = patch_prompt(obs)
        raw     = self._call(prompt)
        payload = _parse_json(raw)
        return PatchAction(
            patch     = payload.get("patch", {}),
            rationale = payload.get("rationale", ""),
        )

    def get_prompt(self, obs: Observation) -> str:
        return patch_prompt(obs)


# ─────────────────────────────────────────────────────────────────
# Orchestrator — runs a complete episode via WebSocket
# ─────────────────────────────────────────────────────────────────

class RAGDiagnosticOrchestrator:
    """
    Connects two agents to the live environment server and runs one episode.

    Usage:
        orch  = RAGDiagnosticOrchestrator(a1, a2, base_url="ws://localhost:8000/ws")
        result = asyncio.run(orch.run_episode("chunking_error_001"))
    """

    def __init__(
        self,
        diagnostic_agent: DiagnosticAgent,
        patch_agent: PatchAgent,
        base_url: str = "ws://localhost:8000/ws",
    ):
        self.a1  = diagnostic_agent
        self.a2  = patch_agent
        self.url = base_url

    async def run_episode(self, task_id: str | None = None) -> dict[str, Any]:
        async with RAGDiagnosticClient(self.url) as client:
            obs = await client.reset(task_id)

            # Agent 1: diagnose
            a1_action = self.a1.act(obs)
            obs       = await client.step(a1_action)
            r1        = obs.reward

            # Agent 2: patch
            a2_action = self.a2.act(obs)
            obs       = await client.step(a2_action)

        return {
            "task_id":   obs.task_id,
            "difficulty": obs.difficulty,
            "agent1": {
                "root_cause": a1_action.root_cause,
                "confidence": a1_action.confidence,
                "step_reward": r1,
            },
            "agent2": {
                "patch":       a2_action.patch,
                "step_reward": obs.reward,
            },
            "episode_total_reward": obs.info.get("episode_total_reward", 0.0),
            "ground_truth_patch":   obs.info.get("ground_truth_patch", {}),
            "terminated":           obs.terminated,
        }
