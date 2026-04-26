"""
OpenEnv-compatible Pydantic models for RAG Diagnostic Gym.

The WebSocket contract expects:
  Action   → sent by the agent (client → server)
  Observation → returned by the environment (server → client)
"""
from __future__ import annotations
from typing import Annotated, Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, RootModel


# ─────────────────────────────────────────────────────────────────
# Action  (one per agent turn)
# ─────────────────────────────────────────────────────────────────

class DiagnoseAction(BaseModel):
    """Agent 1 — DiagnosticAgent."""
    action_type: Literal["diagnose"] = "diagnose"
    root_cause: str = Field(..., description="snake_case root-cause identifier")
    explanation: str = Field(..., description="2-4 sentence causal explanation")
    confidence: float = Field(..., ge=0.0, le=1.0)


class PatchAction(BaseModel):
    """Agent 2 — PatchAgent."""
    action_type: Literal["patch"] = "patch"
    patch: Dict[str, Any] = Field(..., description="config key→value pairs to apply")
    rationale: str = Field(default="", description="1-2 sentence justification")


class ResetAction(BaseModel):
    """Reset the environment to a new or specified task."""
    action_type: Literal["reset"] = "reset"
    task_id: Optional[str] = Field(
        default=None,
        description="One of: chunking_error_001 | embedding_mismatch_001 | hallucination_retrieval_001. "
                    "If omitted, sampled uniformly.",
    )


# Union type for the WS dispatcher
Action = DiagnoseAction | PatchAction | ResetAction

# OpenEnv's create_fastapi_app expects a pydantic model class with model_validate.
StepAction = Annotated[DiagnoseAction | PatchAction, Field(discriminator="action_type")]


class EnvStepAction(RootModel[StepAction]):
    """Pydantic wrapper for discriminated step actions (diagnose / patch)."""
    pass


# ─────────────────────────────────────────────────────────────────
# Observation  (returned after every action)
# ─────────────────────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    diagnosis_score:   Optional[float] = None
    patch_f1:          Optional[float] = None
    faithfulness_delta: Optional[float] = None
    difficulty_multiplier: float = 1.0
    scalar_reward:     float = 0.0


class Observation(BaseModel):
    task_id:    str
    difficulty: Literal["easy", "medium", "hard"]
    step:       int           # 0 = awaiting diagnose, 1 = awaiting patch, 2 = done
    symptoms:   Dict[str, Any]
    diagnosis:  Optional[Dict[str, Any]] = None   # populated after step 0
    reward:     float = 0.0
    reward_breakdown: Optional[RewardBreakdown] = None
    # OpenEnv server internals expect `done`; keep `terminated` for gym semantics.
    done:       bool = False
    terminated: bool = False
    info:       Dict[str, Any] = Field(default_factory=dict)
