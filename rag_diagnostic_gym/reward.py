"""
Reward model — three-component scoring for RAG Diagnostic Gym.

Components
──────────
  R_diag  (0.35) — root-cause identification quality
  R_patch (0.45) — config patch correctness (F1 over key-value pairs)
  R_faith (0.20) — simulated post-patch faithfulness improvement

All components in [0,1].  Scalar = weight-sum × difficulty_multiplier.

TRL/GRPO-compatible reward function wrappers are at the bottom.
"""
from __future__ import annotations

import json
import re
from typing import Any

from openenv.core.rubrics import Rubric
from rag_diagnostic_gym.tasks import TASKS, DIFFICULTY_MULTIPLIER

W = {"diag": 0.35, "patch": 0.45, "faith": 0.20}

# Maximum difficulty multiplier — used to normalize rewards to [0, 1]
MAX_MULTIPLIER = max(DIFFICULTY_MULTIPLIER.values())   # 1.6


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _safe_parse(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group()) if m else {}


def _patch_f1(predicted: dict, correct: dict) -> tuple[float, float, float]:
    if not predicted:
        return 0.0, 0.0, 0.0
    tp = sum(1 for k, v in correct.items() if predicted.get(k) == v)
    p = tp / len(predicted)
    r = tp / len(correct)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# ─────────────────────────────────────────────────────────────────
# Component scorers
# ─────────────────────────────────────────────────────────────────

def score_diagnosis(
    root_cause: str,
    explanation: str,
    confidence: float,
    task_id: str,
) -> tuple[float, dict]:
    task = TASKS[task_id]
    gt   = task["root_cause"]
    pred = root_cause.lower().strip().replace("-", "_")

    # Base: exact match → 1.0, partial keyword → 0.5, miss → 0.0
    if pred == gt:
        base = 1.0
    elif any(tok in pred for tok in gt.split("_")):
        base = 0.5
    else:
        base = 0.0

    # Explanation keyword coverage (max +0.15)
    gt_words   = set(task["root_cause_explanation"].lower().split())
    pred_words = set(explanation.lower().split())
    expl_bonus = min(len(pred_words & gt_words) / 20, 0.15)

    # Calibration: overconfident wrong → penalty; correct + confident → bonus
    if base == 1.0 and confidence >= 0.8:
        cal = +0.10
    elif base == 0.0 and confidence >= 0.8:
        cal = -0.15
    elif base == 0.0 and confidence < 0.4:
        cal = +0.05   # correctly uncertain
    else:
        cal = 0.0

    raw = min(base + expl_bonus + cal, 1.0)
    return raw, {
        "base_match": base,
        "explanation_bonus": round(expl_bonus, 3),
        "calibration": round(cal, 3),
        "diagnosis_score": round(raw, 3),
    }


def score_patch(submitted: dict, task_id: str) -> tuple[float, dict]:
    task      = TASKS[task_id]
    correct   = task["correct_patch"]
    distractors = {k for d in task.get("patch_alternatives", []) for k in d if k not in correct}

    p, r, f1 = _patch_f1(submitted, correct)
    penalty  = 0.08 * sum(1 for k in submitted if k in distractors)
    score    = max(0.0, f1 - penalty)
    if submitted == correct:
        score = min(score + 0.05, 1.0)   # exact-match bonus

    return score, {
        "patch_precision": round(p, 3),
        "patch_recall":    round(r, 3),
        "patch_f1":        round(f1, 3),
        "distractor_penalty": round(penalty, 3),
        "patch_score": round(score, 3),
    }


def score_faithfulness(submitted: dict, task_id: str) -> float:
    task    = TASKS[task_id]
    correct = task["correct_patch"]
    _, r, _ = _patch_f1(submitted, correct)
    baseline = task["observable_state"].get("faithfulness_score", 0.3)
    target   = task["expected_post_patch"].get("faithfulness_score", 0.85)
    return round(baseline + r * (target - baseline), 3)


# ─────────────────────────────────────────────────────────────────
# Composite
# ─────────────────────────────────────────────────────────────────

def composite_reward(
    task_id: str,
    diag: dict,   # {root_cause, explanation, confidence}
    patch: dict,  # {patch: {...}}
) -> tuple[float, dict]:
    task = TASKS[task_id]
    mult = DIFFICULTY_MULTIPLIER[task["difficulty"]]

    d_score, d_info = score_diagnosis(
        diag.get("root_cause", ""),
        diag.get("explanation", ""),
        float(diag.get("confidence", 0.5)),
        task_id,
    )
    p_score, p_info = score_patch(patch.get("patch", {}), task_id)
    f_score         = score_faithfulness(patch.get("patch", {}), task_id)

    scalar = mult * (W["diag"] * d_score + W["patch"] * p_score + W["faith"] * f_score)

    return scalar, {
        "task_id": task_id,
        "difficulty": task["difficulty"],
        "difficulty_multiplier": mult,
        **d_info, **p_info,
        "faithfulness_score": f_score,
        "scalar_reward": round(scalar, 4),
    }


# ─────────────────────────────────────────────────────────────────
# Rubric breakdown (used by the environment server)
# ─────────────────────────────────────────────────────────────────

def apply_rubrics(
    task_id: str,
    diag_payload: dict,
    patch_payload: dict,
) -> dict:
    """Return a composable rubric breakdown for a full episode.

    Called by the environment after both steps are complete.
    ``diag_payload`` is the serialized DiagnoseAction and
    ``patch_payload`` is the serialized PatchAction.
    """
    d_score, d_info = score_diagnosis(
        diag_payload.get("root_cause", ""),
        diag_payload.get("explanation", ""),
        float(diag_payload.get("confidence", 0.5)),
        task_id,
    )
    submitted_patch = patch_payload.get("patch", {})
    p_score, p_info = score_patch(submitted_patch, task_id)
    f_score         = score_faithfulness(submitted_patch, task_id)

    task = TASKS[task_id]
    mult = DIFFICULTY_MULTIPLIER[task["difficulty"]]

    return {
        "diagnosis": {
            "weight": W["diag"],
            "raw_score": round(d_score, 4),
            "weighted": round(W["diag"] * d_score, 4),
            **d_info,
        },
        "patch": {
            "weight": W["patch"],
            "raw_score": round(p_score, 4),
            "weighted": round(W["patch"] * p_score, 4),
            **p_info,
        },
        "faithfulness": {
            "weight": W["faith"],
            "raw_score": round(f_score, 4),
            "weighted": round(W["faith"] * f_score, 4),
        },
        "difficulty_multiplier": mult,
        "composite_raw": round(
            mult * (W["diag"] * d_score + W["patch"] * p_score + W["faith"] * f_score), 4
        ),
        "composite_normalized": round(
            min(1.0, mult * (W["diag"] * d_score + W["patch"] * p_score + W["faith"] * f_score) / MAX_MULTIPLIER), 4
        ),
    }


# ─────────────────────────────────────────────────────────────────
# OpenEnv composable rubrics
# ─────────────────────────────────────────────────────────────────

class DiagnosisRubric(Rubric):
    """Scores diagnosis quality for DiagnoseAction observations."""

    def forward(self, action: Any, observation: Any) -> float:
        if getattr(action, "action_type", None) != "diagnose":
            return 0.0
        task_id = observation.task_id
        score, _ = score_diagnosis(
            getattr(action, "root_cause", ""),
            getattr(action, "explanation", ""),
            float(getattr(action, "confidence", 0.5)),
            task_id,
        )
        mult = DIFFICULTY_MULTIPLIER[TASKS[task_id]["difficulty"]]
        return min(1.0, (mult * W["diag"] * score) / MAX_MULTIPLIER)


class PatchRubric(Rubric):
    """Scores patch correctness (key-value F1 + anti-distractor penalties)."""

    def forward(self, action: Any, observation: Any) -> float:
        if getattr(action, "action_type", None) != "patch":
            return 0.0
        task_id = observation.task_id
        patch_payload = getattr(action, "patch", {}) or {}
        score, _ = score_patch(patch_payload, task_id)
        mult = DIFFICULTY_MULTIPLIER[TASKS[task_id]["difficulty"]]
        return min(1.0, (mult * W["patch"] * score) / MAX_MULTIPLIER)


class FaithfulnessRubric(Rubric):
    """Scores expected downstream faithfulness lift from the proposed patch."""

    def forward(self, action: Any, observation: Any) -> float:
        if getattr(action, "action_type", None) != "patch":
            return 0.0
        task_id = observation.task_id
        patch_payload = getattr(action, "patch", {}) or {}
        score = score_faithfulness(patch_payload, task_id)
        mult = DIFFICULTY_MULTIPLIER[TASKS[task_id]["difficulty"]]
        return min(1.0, (mult * W["faith"] * score) / MAX_MULTIPLIER)


class CompositeDiagnosticRubric(Rubric):
    """
    Composes diagnosis, patch, and faithfulness into one anti-hacking reward.

    The rubric returns step-local reward for the current action type:
    - DiagnoseAction -> diagnosis rubric only
    - PatchAction -> patch rubric + faithfulness rubric
    """

    def __init__(self):
        super().__init__()
        self.diagnosis = DiagnosisRubric()
        self.patch = PatchRubric()
        self.faithfulness = FaithfulnessRubric()

    def forward(self, action: Any, observation: Any) -> float:
        atype = getattr(action, "action_type", None)
        if atype == "diagnose":
            return float(self.diagnosis(action, observation))
        if atype == "patch":
            return float(self.patch(action, observation) + self.faithfulness(action, observation))
        return 0.0


# ─────────────────────────────────────────────────────────────────
# TRL / GRPO reward function wrappers
# ─────────────────────────────────────────────────────────────────

def reward_fn_agent1(
    prompts: list[str],
    completions: list[str],
    task_ids: list[str],
    **_,
) -> list[float]:
    """GRPO reward for DiagnosticAgent."""
    rewards = []
    for comp, tid in zip(completions, task_ids):
        try:
            parsed = _safe_parse(comp)
            s, _   = score_diagnosis(
                parsed.get("root_cause", ""),
                parsed.get("explanation", ""),
                float(parsed.get("confidence", 0.5)),
                tid,
            )
            rewards.append(s * DIFFICULTY_MULTIPLIER[TASKS[tid]["difficulty"]])
        except Exception:
            rewards.append(0.0)
    return rewards


def reward_fn_agent2(
    prompts: list[str],
    completions: list[str],
    task_ids: list[str],
    diag_outputs: list[dict],
    **_,
) -> list[float]:
    """GRPO reward for PatchAgent (patch + faithfulness)."""
    rewards = []
    for comp, tid, diag in zip(completions, task_ids, diag_outputs):
        try:
            parsed  = _safe_parse(comp)
            p_score, _ = score_patch(parsed.get("patch", {}), tid)
            f_score    = score_faithfulness(parsed.get("patch", {}), tid)
            mult       = DIFFICULTY_MULTIPLIER[TASKS[tid]["difficulty"]]
            rewards.append(mult * (W["patch"] * p_score + W["faith"] * f_score))
        except Exception:
            rewards.append(0.0)
    return rewards
