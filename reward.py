"""
Reward model — three-component scoring for RAG Diagnostic Gym.

ALL rewards are normalized to [0, 1] before the difficulty multiplier.
The difficulty multiplier (1.0 / 1.3 / 1.6) is applied AFTER normalization
so the final scalar always lies in [0, 1] when divided by max multiplier (1.6).

Components
──────────
  R_diag  (0.35) — root-cause identification quality
  R_patch (0.45) — config patch correctness (F1 over key-value pairs)
  R_faith (0.20) — simulated post-patch faithfulness improvement

All components in [0, 1].
Scalar = difficulty_multiplier × (w_diag×R_diag + w_patch×R_patch + w_faith×R_faith)
Final reward is then divided by MAX_MULTIPLIER so the episode reward ∈ [0, 1].

TRL/GRPO-compatible reward function wrappers are at the bottom.
"""
from __future__ import annotations

import json
import re
from typing import Any

from rag_diagnostic_gym.tasks import TASKS, DIFFICULTY_MULTIPLIER

W = {"diag": 0.35, "patch": 0.45, "faith": 0.20}
MAX_MULTIPLIER = max(DIFFICULTY_MULTIPLIER.values())   # 1.6  → used to normalize to [0,1]


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _safe_parse(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group()) if m else {}


def _patch_f1(predicted: dict, correct: dict) -> tuple[float, float, float]:
    """Compute precision, recall, F1 over config key-value pairs."""
    if not predicted:
        return 0.0, 0.0, 0.0
    tp = sum(1 for k, v in correct.items() if predicted.get(k) == v)
    p  = tp / len(predicted)
    r  = tp / len(correct)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _normalize(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float to [lo, hi] then map to [0, 1]."""
    return max(0.0, min(1.0, (value - lo) / (hi - lo))) if hi > lo else 0.0


# ─────────────────────────────────────────────────────────────────
# Component scorers  (each returns float ∈ [0, 1])
# ─────────────────────────────────────────────────────────────────

def score_diagnosis(
    root_cause: str,
    explanation: str,
    confidence: float,
    task_id: str,
) -> tuple[float, dict]:
    """
    Returns (score ∈ [0,1], breakdown_dict).

    Sub-components
    ──────────────
    base_match     0.0 / 0.5 / 1.0  — exact / partial-keyword / miss
    expl_bonus     [0, 0.15]        — keyword overlap with ground-truth explanation
    calibration    [-0.15, +0.10]   — penalise confident-but-wrong; reward uncertainty
    """
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
    expl_bonus = min(len(pred_words & gt_words) / max(len(gt_words), 1) * 0.30, 0.15)

    # Calibration bonus / penalty
    if base == 1.0 and confidence >= 0.8:
        cal = +0.10
    elif base == 0.0 and confidence >= 0.8:
        cal = -0.15          # confidently wrong
    elif base == 0.0 and confidence < 0.4:
        cal = +0.05          # correctly uncertain about a wrong guess
    else:
        cal = 0.0

    raw = float(max(0.0, min(1.0, base + expl_bonus + cal)))
    return raw, {
        "base_match":        base,
        "explanation_bonus": round(expl_bonus, 4),
        "calibration":       round(cal, 4),
        "diagnosis_score":   round(raw, 4),
    }


def score_patch(submitted: dict, task_id: str) -> tuple[float, dict]:
    """
    Returns (score ∈ [0,1], breakdown_dict).

    Sub-components
    ──────────────
    patch_f1           F1 over (key, value) pairs vs ground truth
    distractor_penalty 0.08 per distractor key included
    exact_bonus        +0.05 if the submitted patch equals the ground truth exactly
    """
    task        = TASKS[task_id]
    correct     = task["correct_patch"]
    distractors = {k for d in task.get("patch_alternatives", []) for k in d if k not in correct}

    p, r, f1 = _patch_f1(submitted, correct)
    penalty  = 0.08 * sum(1 for k in submitted if k in distractors)
    score    = max(0.0, f1 - penalty)
    if submitted == correct:
        score = min(score + 0.05, 1.0)

    return float(score), {
        "patch_precision":     round(p, 4),
        "patch_recall":        round(r, 4),
        "patch_f1":            round(f1, 4),
        "distractor_penalty":  round(penalty, 4),
        "patch_score":         round(score, 4),
    }


def score_faithfulness(submitted: dict, task_id: str) -> float:
    """
    Simulate the post-patch faithfulness improvement as a fraction ∈ [0, 1].
    The score is proportional to patch recall: how many ground-truth keys were
    fixed correctly.  Returns a float ∈ [0, 1].
    """
    task    = TASKS[task_id]
    correct = task["correct_patch"]
    _, r, _ = _patch_f1(submitted, correct)
    # Interpolate between baseline and target faithfulness, then normalize to [0,1]
    baseline = task["observable_state"].get("faithfulness_score", 0.3)
    target   = task["expected_post_patch"].get("faithfulness_score", 0.85)
    raw_faith = baseline + r * (target - baseline)
    # Normalize: faithfulness is already on roughly [0,1]; clamp to be safe
    return round(float(max(0.0, min(1.0, raw_faith))), 4)


# ─────────────────────────────────────────────────────────────────
# Composite — final scalar ∈ [0, 1]
# ─────────────────────────────────────────────────────────────────

def composite_reward(
    task_id: str,
    diag: dict,    # {root_cause, explanation, confidence}
    patch: dict,   # {patch: {...}}
) -> tuple[float, dict]:
    """
    Compute the composite episode reward.

    Returns
    -------
    scalar ∈ [0, 1]   — difficulty-weighted, then divided by MAX_MULTIPLIER
    breakdown dict
    """
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

    weighted = mult * (W["diag"] * d_score + W["patch"] * p_score + W["faith"] * f_score)
    # Normalize so the max achievable score is 1.0
    scalar = float(min(1.0, weighted / MAX_MULTIPLIER))

    return scalar, {
        "task_id":              task_id,
        "difficulty":           task["difficulty"],
        "difficulty_multiplier": mult,
        **d_info, **p_info,
        "faithfulness_score":   f_score,
        "weighted_sum":         round(weighted, 4),
        "scalar_reward":        round(scalar, 4),   # always ∈ [0, 1]
    }


# ─────────────────────────────────────────────────────────────────
# Rubric helpers — composable, OpenEnv-style
# ─────────────────────────────────────────────────────────────────

class DiagnosisRubric:
    """Composable rubric for Agent 1 (DiagnosticAgent)."""
    weight = W["diag"]

    @staticmethod
    def score(root_cause: str, explanation: str, confidence: float, task_id: str) -> float:
        s, _ = score_diagnosis(root_cause, explanation, confidence, task_id)
        return s   # ∈ [0, 1]


class PatchRubric:
    """Composable rubric for Agent 2 (PatchAgent) — patch correctness."""
    weight = W["patch"]

    @staticmethod
    def score(submitted: dict, task_id: str) -> float:
        s, _ = score_patch(submitted, task_id)
        return s   # ∈ [0, 1]


class FaithfulnessRubric:
    """Composable rubric — simulated faithfulness improvement."""
    weight = W["faith"]

    @staticmethod
    def score(submitted: dict, task_id: str) -> float:
        return score_faithfulness(submitted, task_id)   # ∈ [0, 1]


def apply_rubrics(task_id: str, diag: dict, patch: dict) -> dict[str, float]:
    """
    Apply all three rubrics and return a dict of component scores ∈ [0, 1].
    The composite is also normalized to [0, 1].
    """
    d = DiagnosisRubric.score(
        diag.get("root_cause", ""), diag.get("explanation", ""),
        float(diag.get("confidence", 0.5)), task_id,
    )
    p = PatchRubric.score(patch.get("patch", {}), task_id)
    f = FaithfulnessRubric.score(patch.get("patch", {}), task_id)
    mult = DIFFICULTY_MULTIPLIER[TASKS[task_id]["difficulty"]]
    composite = float(min(1.0, mult * (W["diag"] * d + W["patch"] * p + W["faith"] * f) / MAX_MULTIPLIER))
    return {
        "diagnosis":    round(d, 4),
        "patch":        round(p, 4),
        "faithfulness": round(f, 4),
        "composite":    round(composite, 4),
    }


# ─────────────────────────────────────────────────────────────────
# TRL / GRPO reward function wrappers  (return list[float] ∈ [0,1])
# ─────────────────────────────────────────────────────────────────

def reward_fn_agent1(
    prompts: list[str],
    completions: list[str],
    task_ids: list[str],
    **_,
) -> list[float]:
    """
    GRPO reward for DiagnosticAgent.
    Returns list of floats ∈ [0, 1].
    """
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
            mult = DIFFICULTY_MULTIPLIER[TASKS[tid]["difficulty"]]
            # Normalize: diagnosis component max = mult × W["diag"]; divide by MAX_MULTIPLIER
            rewards.append(float(min(1.0, mult * W["diag"] * s / MAX_MULTIPLIER)))
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
    """
    GRPO reward for PatchAgent (patch + faithfulness).
    Returns list of floats ∈ [0, 1].
    """
    rewards = []
    for comp, tid, diag in zip(completions, task_ids, diag_outputs):
        try:
            parsed  = _safe_parse(comp)
            p_score, _ = score_patch(parsed.get("patch", {}), tid)
            f_score    = score_faithfulness(parsed.get("patch", {}), tid)
            mult       = DIFFICULTY_MULTIPLIER[TASKS[tid]["difficulty"]]
            raw = mult * (W["patch"] * p_score + W["faith"] * f_score)
            rewards.append(float(min(1.0, raw / MAX_MULTIPLIER)))
        except Exception:
            rewards.append(0.0)
    return rewards
