"""
rag_diagnostic_gym.train
═════════════════════════
All training utilities for RAG Diagnostic Gym — imported directly by the
Colab notebook, so logic lives here once and nowhere else.

Reward guarantees
─────────────────
  ALL reward functions return floats ∈ [0, 1].
  The difficulty multiplier is baked in and then divided by MAX_MULTIPLIER
  (= 1.6) so that the hardest task's perfect score is exactly 1.0.

Exports
───────
  SYSTEM_PROMPT_AGENT1        system prompt for DiagnosticAgent
  SYSTEM_PROMPT_AGENT2        system prompt for PatchAgent
  rollout_once                run one full two-step episode
  format_symptoms             format obs.symptoms for the prompt
  format_history              render multi-turn conversation history
  parse_json_output           robust JSON extractor from LLM completions
  apply_chat_template         wrap prompt in model chat format
  reward_total                GRPO reward func — composite scalar ∈ [0, 1]
  reward_diagnosis            GRPO reward func — diagnosis component ∈ [0, 1]
  reward_patch                GRPO reward func — patch component ∈ [0, 1]
  plot_rewards                save reward-curve PNG from trainer log history
  patch_trl_vllm_compat       monkey-patch TRL/vLLM version mismatches
"""
from __future__ import annotations

import csv
import json
import re
import textwrap
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────
# System Prompts
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_AGENT1 = textwrap.dedent("""
    You are a senior ML infrastructure engineer on-call.
    You will be given observable metrics and logs from a broken RAG pipeline.

    Your job (Step 1 of 2):
      Identify the single root cause that explains ALL the symptoms.

    Think step-by-step inside <think>...</think> tags, then output ONLY valid
    JSON — no markdown fences, no extra text:

    {
      "root_cause": "<snake_case_identifier>",
      "explanation": "<2-4 sentences explaining why this root cause produces all symptoms>",
      "confidence": <float 0.0-1.0>
    }

    Known root cause identifiers:
      chunk_size_too_large
      embedding_model_mismatch
      query_expansion_semantic_drift_no_reranker
""").strip()

SYSTEM_PROMPT_AGENT2 = textwrap.dedent("""
    You are a senior ML infrastructure engineer on-call.
    Agent 1 has diagnosed the root cause. Your job (Step 2 of 2):
      Emit the minimal config patch that resolves the issue.

    Think step-by-step inside <think>...</think> tags, then output ONLY valid
    JSON — no markdown fences, no extra text:

    {
      "patch": {
        "<config_key>": <corrected_value>
      },
      "rationale": "<1-2 sentences>"
    }

    Only include config keys that must change. Do not include distractor keys.
""").strip()


# ─────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────

def format_symptoms(symptoms: dict) -> str:
    """Pretty-print observable symptoms for inclusion in the prompt."""
    return json.dumps(symptoms, indent=2)


def format_history(history: list[dict]) -> str:
    """
    Render a conversation history list as a readable string.
    Each entry: {"role": "agent1"|"env", "content": str}
    """
    lines = []
    for turn in history:
        role    = turn.get("role", "unknown").upper()
        content = turn.get("content", "")
        lines.append(f"[{role}]\n{content}")
    return "\n\n".join(lines)


def apply_chat_template(tokenizer, system_prompt: str, user_content: str) -> str:
    """
    Wrap system + user content in the model's chat template.
    Falls back to a plain concatenation if the tokenizer has no template.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"{system_prompt}\n\nUser: {user_content}\nAssistant:"


def parse_json_output(text: str) -> dict:
    """
    Robust JSON extractor.
    Handles: markdown fences, <think> tags, leading/trailing noise.
    """
    # Strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Extract first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


_safe_parse = parse_json_output    # alias used internally


# ─────────────────────────────────────────────────────────────────
# Rollout — one full two-step episode
# ─────────────────────────────────────────────────────────────────

def rollout_once(
    trainer,
    env_url: str,
    tokenizer,
    task_id: str | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    Run one complete episode (diagnose → patch) against the live env server.

    All rewards in the returned dict are ∈ [0, 1].

    Returns
    -------
    dict with keys:
      prompt_ids          list[int]
      completion_ids      list[int]
      logprobs            list[float]
      total_reward        float ∈ [0, 1]
      diagnosis_reward    float ∈ [0, 1]
      patch_reward        float ∈ [0, 1]
      faithfulness_reward float ∈ [0, 1]
      task_id             str
      difficulty          str
      agent1_output       dict
      agent2_output       dict
      patch_submitted     dict
      ground_truth_patch  dict
    """
    import asyncio
    import torch
    import torch.nn.functional as F

    from rag_diagnostic_gym.client import RAGDiagnosticClient
    from rag_diagnostic_gym.models import DiagnoseAction, PatchAction
    from rag_diagnostic_gym.reward import (
        MAX_MULTIPLIER, W,
        score_diagnosis, score_patch, score_faithfulness,
    )
    from rag_diagnostic_gym.tasks import TASKS, DIFFICULTY_MULTIPLIER

    async def _run() -> dict:
        ws_url = env_url.replace("https://", "wss://").replace("http://", "ws://")
        if not ws_url.endswith("/ws"):
            ws_url = ws_url.rstrip("/") + "/ws"

        async with RAGDiagnosticClient(ws_url) as client:
            obs = await client.reset(task_id)
            tid = obs.task_id

            # ── Build Agent 1 prompt ──────────────────────────────
            user_content = (
                f"## Pipeline Symptoms\n```json\n{format_symptoms(obs.symptoms)}\n```"
            )
            prompt_text = apply_chat_template(tokenizer, SYSTEM_PROMPT_AGENT1, user_content)

            # ── Generate Agent 1 completion ───────────────────────
            inputs    = tokenizer(prompt_text, return_tensors="pt")
            prompt_ids = inputs["input_ids"][0].tolist()

            with torch.no_grad():
                out = trainer.model.generate(
                    **{k: v.to(trainer.model.device) for k, v in inputs.items()},
                    max_new_tokens=384,
                    temperature=0.7,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            new_ids    = out.sequences[0][len(prompt_ids):].tolist()
            a1_text    = tokenizer.decode(new_ids, skip_special_tokens=True)
            a1_payload = parse_json_output(a1_text)

            # Per-token log-probabilities
            logprobs = []
            if hasattr(out, "scores") and out.scores:
                for step_scores, tok_id in zip(out.scores, new_ids):
                    lp = F.log_softmax(step_scores[0], dim=-1)
                    logprobs.append(lp[tok_id].item())

            # ── Step env with Agent 1 action ──────────────────────
            obs = await client.step(DiagnoseAction(
                root_cause  = a1_payload.get("root_cause", "unknown"),
                explanation = a1_payload.get("explanation", ""),
                confidence  = float(a1_payload.get("confidence", 0.5)),
            ))

            # ── Build Agent 2 prompt ──────────────────────────────
            user2 = (
                f"## Pipeline Symptoms\n```json\n{format_symptoms(obs.symptoms)}\n```\n\n"
                f"## Agent 1 Diagnosis\n```json\n{json.dumps(obs.diagnosis, indent=2)}\n```"
            )
            prompt2_text = apply_chat_template(tokenizer, SYSTEM_PROMPT_AGENT2, user2)
            inputs2      = tokenizer(prompt2_text, return_tensors="pt")

            with torch.no_grad():
                out2 = trainer.model.generate(
                    **{k: v.to(trainer.model.device) for k, v in inputs2.items()},
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                )
            a2_text    = tokenizer.decode(
                out2[0][inputs2["input_ids"].shape[-1]:], skip_special_tokens=True
            )
            a2_payload = parse_json_output(a2_text)

            # ── Step env with Agent 2 action ──────────────────────
            obs = await client.step(PatchAction(
                patch     = a2_payload.get("patch", {}),
                rationale = a2_payload.get("rationale", ""),
            ))

            # ── Score (all ∈ [0, 1]) ──────────────────────────────
            mult = DIFFICULTY_MULTIPLIER[TASKS[tid]["difficulty"]]
            d_score, _ = score_diagnosis(
                a1_payload.get("root_cause", ""),
                a1_payload.get("explanation", ""),
                float(a1_payload.get("confidence", 0.5)),
                tid,
            )
            p_score, _ = score_patch(a2_payload.get("patch", {}), tid)
            f_score    = score_faithfulness(a2_payload.get("patch", {}), tid)

            # Normalize each component to [0, 1]
            diag_r  = float(min(1.0, mult * W["diag"]  * d_score / MAX_MULTIPLIER))
            patch_r = float(min(1.0, mult * W["patch"] * p_score / MAX_MULTIPLIER))
            faith_r = float(min(1.0, mult * W["faith"] * f_score / MAX_MULTIPLIER))
            total   = float(min(1.0, mult * (
                W["diag"] * d_score + W["patch"] * p_score + W["faith"] * f_score
            ) / MAX_MULTIPLIER))

            return {
                "prompt_ids":          prompt_ids,
                "completion_ids":      new_ids,
                "logprobs":            logprobs,
                "total_reward":        round(total, 4),
                "diagnosis_reward":    round(diag_r, 4),
                "patch_reward":        round(patch_r, 4),
                "faithfulness_reward": round(faith_r, 4),
                "task_id":             tid,
                "difficulty":          obs.difficulty,
                "agent1_output":       a1_payload,
                "agent2_output":       a2_payload,
                "patch_submitted":     a2_payload.get("patch", {}),
                "ground_truth_patch":  obs.info.get("ground_truth_patch", {}),
            }

    return asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────
# TRL / GRPO reward function wrappers  (all return list[float] ∈ [0,1])
# ─────────────────────────────────────────────────────────────────

def reward_total(
    prompts: list[str],
    completions: list[str],
    task_ids: list[str] | None = None,
    diag_outputs: list[dict] | None = None,
    **_,
) -> list[float]:
    """
    Composite reward ∈ [0, 1] — used as primary GRPO reward func for Agent 1.
    """
    from rag_diagnostic_gym.reward import reward_fn_agent1
    return reward_fn_agent1(prompts, completions, task_ids=task_ids or [])


def reward_diagnosis(
    prompts: list[str],
    completions: list[str],
    task_ids: list[str] | None = None,
    **_,
) -> list[float]:
    """
    Diagnosis-only reward ∈ [0, 1] — logged separately for interpretability.
    """
    from rag_diagnostic_gym.reward import score_diagnosis, W, MAX_MULTIPLIER
    from rag_diagnostic_gym.tasks import TASKS, DIFFICULTY_MULTIPLIER
    results = []
    for comp, tid in zip(completions, task_ids or []):
        try:
            p    = _safe_parse(comp)
            s, _ = score_diagnosis(
                p.get("root_cause", ""), p.get("explanation", ""),
                float(p.get("confidence", 0.5)), tid,
            )
            mult = DIFFICULTY_MULTIPLIER[TASKS[tid]["difficulty"]]
            results.append(float(min(1.0, mult * W["diag"] * s / MAX_MULTIPLIER)))
        except Exception:
            results.append(0.0)
    return results


def reward_patch(
    prompts: list[str],
    completions: list[str],
    task_ids: list[str] | None = None,
    **_,
) -> list[float]:
    """
    Patch-only reward ∈ [0, 1] — logged separately for interpretability.
    """
    from rag_diagnostic_gym.reward import score_patch, score_faithfulness, W, MAX_MULTIPLIER
    from rag_diagnostic_gym.tasks import TASKS, DIFFICULTY_MULTIPLIER
    results = []
    for comp, tid in zip(completions, task_ids or []):
        try:
            p       = _safe_parse(comp)
            ps, _   = score_patch(p.get("patch", {}), tid)
            fs      = score_faithfulness(p.get("patch", {}), tid)
            mult    = DIFFICULTY_MULTIPLIER[TASKS[tid]["difficulty"]]
            raw     = mult * (W["patch"] * ps + W["faith"] * fs)
            results.append(float(min(1.0, raw / MAX_MULTIPLIER)))
        except Exception:
            results.append(0.0)
    return results


# ─────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────

def plot_rewards(
    trainer_a1=None,
    trainer_a2=None,
    csv_path: str | Path | None = None,
    out_path: str | Path = "plots/reward_curves.png",
) -> None:
    """
    Save a reward-curve PNG from trainer log history or from a CSV file.

    Trainer log history keys:     rewards/mean (or reward/mean)
    CSV columns (legacy):         episode, total_reward, diagnosis_reward, patch_reward

    The y-axis is labelled [0, 1] since all rewards are normalized.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def smooth(arr, w=5):
        if len(arr) < w:
            return np.array(arr)
        return np.convolve(arr, np.ones(w) / w, mode="valid")

    def extract_from_trainer(trainer, key="rewards/mean"):
        """Try multiple key variants to handle TRL version differences."""
        # Explicit key candidates (old + new naming conventions)
        candidates = [
            key,
            "reward/mean",
            "train/rewards/mean",
            "train/reward/mean",
            # New TRL: rewards are logged per reward-function name
            "rewards/agent1_reward/mean",
            "rewards/agent2_reward/mean",
        ]
        for k in candidates:
            pts = [(e["step"], e[k]) for e in trainer.state.log_history if k in e]
            if pts:
                return pts
        # Fallback: find any key matching rewards/*/mean
        for e in trainer.state.log_history:
            for k in e:
                if k.startswith("rewards/") and k.endswith("/mean"):
                    pts = [(entry["step"], entry[k])
                           for entry in trainer.state.log_history if k in entry]
                    if pts:
                        return pts
        return []

    # ── Case 1: trainer objects available ─────────────────────────
    if trainer_a1 is not None or trainer_a2 is not None:
        trainers = []
        if trainer_a1 is not None:
            trainers.append((trainer_a1, "Agent 1 — DiagnosticAgent (diagnosis reward)", "#534AB7"))
        if trainer_a2 is not None:
            trainers.append((trainer_a2, "Agent 2 — PatchAgent (patch reward)", "#0F6E56"))

        ncols = len(trainers)
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
        if ncols == 1:
            axes = [axes]
        fig.suptitle(
            "RAG Diagnostic Gym — GRPO Training Rewards\n"
            "All rewards normalised to [0, 1]",
            fontsize=13, fontweight="bold",
        )

        for ax, (trainer, label, color) in zip(axes, trainers):
            pts = extract_from_trainer(trainer)
            if pts:
                steps, vals = zip(*pts)
                steps, vals = list(steps), list(vals)
                ax.plot(steps, vals, color=color, alpha=0.3, linewidth=1, label="raw")
                sm     = smooth(vals)
                sm_eps = steps[len(steps) - len(sm):]
                ax.plot(sm_eps, sm, color=color, linewidth=2.5, label="smoothed (w=5)")
            else:
                ax.text(0.5, 0.5, "No log data yet", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="gray")

            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training step", fontsize=10)
            ax.set_ylabel("Reward  [0 → 1]", fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)

    # ── Case 2: CSV file (legacy / offline) ───────────────────────
    elif csv_path is not None:
        csv_path = Path(csv_path)
        episodes, total, diag, patch = [], [], [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                episodes.append(int(row["episode"]))
                total.append(float(row.get("total_reward", 0)))
                diag.append(float(row.get("diagnosis_reward", 0)))
                patch.append(float(row.get("patch_reward", 0)))

        if not episodes:
            print("No data to plot yet.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            "RAG Diagnostic Gym — Training Rewards\nAll rewards ∈ [0, 1]",
            fontsize=13, fontweight="bold",
        )
        for ax, vals, label, color in [
            (axes[0], total, "Total Reward",     "#534AB7"),
            (axes[1], diag,  "Diagnosis Reward",  "#185FA5"),
            (axes[2], patch, "Patch Reward",      "#0F6E56"),
        ]:
            ax.plot(episodes, vals, color=color, alpha=0.3, linewidth=1, label="raw")
            sm     = smooth(vals)
            sm_eps = episodes[len(episodes) - len(sm):]
            ax.plot(sm_eps, sm, color=color, linewidth=2.5, label="smoothed (w=5)")
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Episode", fontsize=10)
            ax.set_ylabel("Reward  [0 → 1]", fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    else:
        raise ValueError("Provide either trainer objects or a csv_path.")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Reward curve saved → {out_path}")


# ─────────────────────────────────────────────────────────────────
# TRL / vLLM compatibility patch
# ─────────────────────────────────────────────────────────────────

def patch_trl_vllm_compat() -> None:
    """
    Apply monkey-patches for known TRL ≥ 0.29 / vLLM ≥ 0.11 API drift.
    Safe to call even when vLLM is not installed (no-ops in that case).
    """
    try:
        import vllm
        from packaging.version import Version as V

        vllm_ver = V(vllm.__version__)

        if vllm_ver >= V("0.11.0"):
            try:
                from vllm import SamplingParams
                _orig_init = SamplingParams.__init__

                def _patched_init(self, *args, **kwargs):
                    if "logprobs" in kwargs and "prompt_logprobs" not in kwargs:
                        kwargs["prompt_logprobs"] = kwargs.pop("logprobs")
                    _orig_init(self, *args, **kwargs)

                SamplingParams.__init__ = _patched_init
            except Exception:
                pass

        print(f"patch_trl_vllm_compat: vLLM {vllm.__version__} — patches applied")

    except ImportError:
        print("patch_trl_vllm_compat: vLLM not installed — no patches needed")
    except Exception as e:
        print(f"patch_trl_vllm_compat: warning — {e}")
