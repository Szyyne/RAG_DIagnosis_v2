"""
GRPO Training Script — RAG Diagnostic Gym
==========================================
Stage 1: Fine-tune DiagnosticAgent (Agent 1) with reward_fn_agent1
Stage 2: Fine-tune PatchAgent      (Agent 2) with reward_fn_agent2

All rewards are normalised to [0, 1] before being fed to GRPOTrainer.

Uses:
  • Unsloth   — 4-bit QLoRA, 2× faster training on free Colab T4
  • TRL       — GRPOTrainer
  • HuggingFace datasets

Install (Colab cell):
  !pip install unsloth trl datasets peft accelerate bitsandbytes -q

Run:
  python training/train_grpo.py --stage 1   # train Agent 1
  python training/train_grpo.py --stage 2   # train Agent 2 (requires Agent 1 checkpoint)
  python training/train_grpo.py --eval      # evaluate both agents
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import threading
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from rag_diagnostic_gym.reward import reward_fn_agent1, reward_fn_agent2, MAX_MULTIPLIER
from rag_diagnostic_gym.tasks import TASKS
from train import plot_rewards
from rag_diagnostic_gym.client import RAGDiagnosticClient
from rag_diagnostic_gym.models import DiagnoseAction, PatchAction
from agents.orchestrator import DiagnosticAgent, PatchAgent, RAGDiagnosticOrchestrator


# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────

BASE_MODEL      = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LEN     = 2048
LORA_RANK       = 16
BATCH_SIZE      = 4
GRAD_ACCUM      = 4
LR              = 2e-5
EPOCHS          = 3
NUM_GENERATIONS = 8    # GRPO group size (completions per prompt)
CKPT_DIR        = Path("./checkpoints")
PLOTS_DIR       = Path("./plots")
ENV_URL         = os.getenv("ENV_WS_URL", "ws://localhost:8000/ws")


# ─────────────────────────────────────────────────────────────────
# CSV Reward Logger
# ─────────────────────────────────────────────────────────────────

_CSV_COLUMNS = ["timestamp", "epoch", "step", "task_id", "difficulty", "reward", "source"]


class RewardCSVLogger:
    """Append-only CSV logger for reward values.

    Thread-safe — safe to call from GRPOTrainer reward functions which may
    run on background threads.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._step = 0
        self._epoch = 0
        # Write header if the file is new or empty
        if not self.path.exists() or self.path.stat().st_size == 0:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(_CSV_COLUMNS)

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def inc_step(self):
        self._step += 1

    def log(self, task_id: str, reward: float, source: str):
        difficulty = TASKS.get(task_id, {}).get("difficulty", "unknown")
        row = [
            datetime.utcnow().isoformat(timespec="seconds"),
            self._epoch,
            self._step,
            task_id,
            difficulty,
            round(reward, 6),
            source,
        ]
        with self._lock:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    def log_batch(self, task_ids: list[str], rewards: list[float], source: str):
        """Log a batch of rewards in one go."""
        self.inc_step()
        for tid, r in zip(task_ids, rewards):
            self.log(tid, r, source)


# Global loggers — initialised lazily by the make_*_reward functions.
_a1_csv_logger: RewardCSVLogger | None = None
_a2_csv_logger: RewardCSVLogger | None = None


def get_a1_logger(path: str | Path | None = None) -> RewardCSVLogger:
    """Get or create the Agent 1 reward CSV logger."""
    global _a1_csv_logger
    if _a1_csv_logger is None:
        _a1_csv_logger = RewardCSVLogger(path or PLOTS_DIR / "agent1_rewards.csv")
    return _a1_csv_logger


def get_a2_logger(path: str | Path | None = None) -> RewardCSVLogger:
    """Get or create the Agent 2 reward CSV logger."""
    global _a2_csv_logger
    if _a2_csv_logger is None:
        _a2_csv_logger = RewardCSVLogger(path or PLOTS_DIR / "agent2_rewards.csv")
    return _a2_csv_logger


def dump_trainer_rewards_to_csv(
    trainer,
    out_path: str | Path,
    agent_name: str = "agent1",
) -> Path:
    """Extract reward data from a GRPOTrainer's log_history and write to CSV.

    This is a fallback for extracting rewards *after* training finishes,
    in case the inline logger was not active.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Find the reward key
    reward_key = None
    for e in trainer.state.log_history:
        for k in e:
            if k.startswith("rewards/") and k.endswith("/mean"):
                reward_key = k
                break
        if reward_key:
            break
    # Fallback to older TRL keys
    if reward_key is None:
        for k in ["rewards/mean", "reward/mean"]:
            if any(k in e for e in trainer.state.log_history):
                reward_key = k
                break

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "reward_mean", "loss", "source"])
        for entry in trainer.state.log_history:
            step = entry.get("step", -1)
            reward = entry.get(reward_key) if reward_key else None
            loss = entry.get("loss")
            if reward is not None or loss is not None:
                writer.writerow([
                    step,
                    round(reward, 6) if reward is not None else "",
                    round(loss, 6) if loss is not None else "",
                    agent_name,
                ])

    print(f"✓ Dumped trainer rewards → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────
# Model utilities
# ─────────────────────────────────────────────────────────────────

def load_base_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL,
        max_seq_length = MAX_SEQ_LEN,
        load_in_4bit   = True,
        dtype          = None,
    )
    return model, tokenizer


def apply_lora(model):
    return FastLanguageModel.get_peft_model(
        model,
        r                          = LORA_RANK,
        target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
        lora_alpha                 = LORA_RANK * 2,
        lora_dropout               = 0.0,
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = 42,
    )


def model_fn_from_weights(model, tokenizer, temperature: float = 0.3):
    """Build a ModelFn callable from a loaded Unsloth model."""
    FastLanguageModel.for_inference(model)

    def _fn(prompt) -> str:
        # Defensive: ensure prompt is always a plain string for the tokenizer
        if not isinstance(prompt, str):
            import json as _json
            if hasattr(prompt, "model_dump"):        # Pydantic model
                prompt = _json.dumps(prompt.model_dump())
            elif hasattr(prompt, "__dict__"):         # arbitrary object
                prompt = str(prompt)
            elif isinstance(prompt, (dict, list)):
                prompt = _json.dumps(prompt)
            else:
                prompt = str(prompt)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens = 256,
                temperature    = temperature,
                do_sample      = True,
            )
        return tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

    return _fn


# ─────────────────────────────────────────────────────────────────
# Dataset builders
# ─────────────────────────────────────────────────────────────────

def _difficulty_copies(task_id: str) -> int:
    """Hard tasks get more copies → proportionally more gradient signal."""
    d = TASKS[task_id]["difficulty"]
    return {"easy": 20, "medium": 35, "hard": 50}[d]


def build_agent1_dataset() -> Dataset:
    """Prompt dataset for DiagnosticAgent (Agent 1)."""
    from agents.orchestrator import DiagnosticAgent, diagnostic_prompt
    from rag_diagnostic_gym.models import Observation

    dummy = DiagnosticAgent(model_fn=lambda p: "{}")
    rows  = []
    for tid, task in TASKS.items():
        obs = Observation(
            task_id    = tid,
            difficulty = task["difficulty"],
            step       = 0,
            symptoms   = task["observable_state"],
        )
        prompt = dummy.get_prompt(obs)
        rows.extend([{"prompt": prompt, "task_id": tid}] * _difficulty_copies(tid))
    return Dataset.from_list(rows)


def build_agent2_dataset(a1_model_fn) -> Dataset:
    """Prompt dataset for PatchAgent (Agent 2) — includes Agent 1 diagnosis."""
    from agents.orchestrator import DiagnosticAgent, PatchAgent, patch_prompt
    from rag_diagnostic_gym.models import Observation

    a1   = DiagnosticAgent(model_fn=a1_model_fn)
    rows = []

    for tid, task in TASKS.items():
        obs_no_diag = Observation(
            task_id    = tid,
            difficulty = task["difficulty"],
            step       = 0,
            symptoms   = task["observable_state"],
        )
        try:
            a1_action = a1.act(obs_no_diag)
        except Exception:
            # Some intermediate checkpoints can emit empty/non-JSON text.
            # Keep dataset construction robust with a conservative fallback.
            a1_action = DiagnoseAction(
                root_cause="unknown",
                explanation="",
                confidence=0.5,
            )
        obs_with_diag = Observation(
            task_id    = tid,
            difficulty = task["difficulty"],
            step       = 1,
            symptoms   = task["observable_state"],
            diagnosis  = a1_action.model_dump(),
        )
        dummy_a2 = PatchAgent(model_fn=lambda p: "{}")
        prompt   = dummy_a2.get_prompt(obs_with_diag)
        rows.extend([{
            "prompt":      prompt,
            "task_id":     tid,
            "diag_output": json.dumps(a1_action.model_dump()),
        }] * _difficulty_copies(tid))

    return Dataset.from_list(rows)


# ─────────────────────────────────────────────────────────────────
# Reward wrappers for GRPOTrainer  (all return list[float] ∈ [0,1])
# ─────────────────────────────────────────────────────────────────

def make_a1_reward(csv_path: str | Path | None = None):
    """Return a GRPO-compatible reward function for Agent 1."""
    logger = get_a1_logger(csv_path)

    def agent1_reward(prompts, completions, **kw):
        completions = _extract_completions(completions)
        tids = kw.get("task_id", [])
        rewards = reward_fn_agent1(prompts, completions, task_ids=tids)
        # Guard: ensure all values are in [0, 1]
        rewards = [float(min(1.0, max(0.0, r))) for r in rewards]
        # Log to CSV
        if tids:
            logger.log_batch(tids, rewards, "offline")
        return rewards
    return agent1_reward


def make_a2_reward(csv_path: str | Path | None = None):
    """Return a GRPO-compatible reward function for Agent 2."""
    logger = get_a2_logger(csv_path)

    def agent2_reward(prompts, completions, **kw):
        completions = _extract_completions(completions)
        diag_outs = [
            _safe_json(d) if isinstance(d, str) else (d or {})
            for d in kw.get("diag_output", ["{}"] * len(completions))
        ]
        tids = kw.get("task_id", [])
        rewards = reward_fn_agent2(
            prompts, completions, task_ids=tids, diag_outputs=diag_outs
        )
        rewards = [float(min(1.0, max(0.0, r))) for r in rewards]
        if tids:
            logger.log_batch(tids, rewards, "offline")
        return rewards
    return agent2_reward


def _extract_completions(completions) -> list[str]:
    """Normalize completions from GRPOTrainer to list[str].

    Newer TRL versions may pass completions as:
      - list[str]                          (plain text)
      - list[list[dict]]                   (chat messages per sample)
      - list[dict] with key 'content'      (single message per sample)
    This helper always returns list[str].
    """
    out: list[str] = []
    for c in completions:
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, dict):
            out.append(c.get("content", str(c)))
        elif isinstance(c, (list, tuple)):
            # list of message dicts — concatenate all 'content' fields
            out.append(" ".join(
                m.get("content", "") if isinstance(m, dict) else str(m)
                for m in c
            ))
        else:
            out.append(str(c))
    return out


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _run_async(coro):
    """Run async code from sync contexts, including Jupyter notebooks."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, float] = {}
    error: dict[str, Exception] = {}

    def _runner():
        try:
            result["value"] = asyncio.run(coro)
        except Exception as e:  # pragma: no cover - defensive path
            error["value"] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()

    if "value" in error:
        raise error["value"]
    return result["value"]


def _safe_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        import re
        stripped = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
        m = re.search(r"\{.*\}", stripped, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                return {}
    return {}


async def _env_reward_a1_once(task_id: str, completion: str, env_url: str) -> float:
    payload = _safe_json(completion)
    action = DiagnoseAction(
        root_cause=payload.get("root_cause", "unknown"),
        explanation=payload.get("explanation", ""),
        confidence=_clip01(float(payload.get("confidence", 0.5))),
    )
    async with RAGDiagnosticClient(env_url) as client:
        await client.reset(task_id)
        obs = await client.step(action)
        return _clip01(float(obs.reward))


async def _env_reward_a2_once(task_id: str, completion: str, diag_output: dict, env_url: str) -> float:
    patch_payload = _safe_json(completion)
    diag_action = DiagnoseAction(
        root_cause=diag_output.get("root_cause", "unknown"),
        explanation=diag_output.get("explanation", ""),
        confidence=_clip01(float(diag_output.get("confidence", 0.5))),
    )
    patch_action = PatchAction(
        patch=patch_payload.get("patch", {}) or {},
        rationale=patch_payload.get("rationale", ""),
    )
    async with RAGDiagnosticClient(env_url) as client:
        await client.reset(task_id)
        await client.step(diag_action)
        obs = await client.step(patch_action)
        return _clip01(float(obs.reward))


def make_a1_reward_online(env_url: str | None = None, csv_path: str | Path | None = None):
    """
    True online reward for Agent 1: GRPO completions are scored by live env steps.
    """
    url = env_url or ENV_URL
    logger = get_a1_logger(csv_path)

    def agent1_reward(prompts, completions, **kw):
        completions = _extract_completions(completions)
        tids = kw.get("task_id", [])
        rewards: list[float] = []
        for comp, tid in zip(completions, tids):
            try:
                rewards.append(_run_async(_env_reward_a1_once(tid, comp, url)))
            except Exception:
                rewards.append(0.0)
        rewards = [_clip01(r) for r in rewards]
        if tids:
            logger.log_batch(tids, rewards, "online")
        return rewards

    return agent1_reward


def make_a2_reward_online(env_url: str | None = None, csv_path: str | Path | None = None):
    """
    True online reward for Agent 2: replay diagnosis + patch in live env and score.
    """
    url = env_url or ENV_URL
    logger = get_a2_logger(csv_path)

    def agent2_reward(prompts, completions, **kw):
        completions = _extract_completions(completions)
        tids = kw.get("task_id", [])
        raw_diags = kw.get("diag_output", ["{}"] * len(completions))
        diag_outs = [
            (_safe_json(d) if isinstance(d, str) else (d or {}))
            for d in raw_diags
        ]
        rewards: list[float] = []
        for comp, tid, diag in zip(completions, tids, diag_outs):
            try:
                rewards.append(_run_async(_env_reward_a2_once(tid, comp, diag, url)))
            except Exception:
                rewards.append(0.0)
        rewards = [_clip01(r) for r in rewards]
        if tids:
            logger.log_batch(tids, rewards, "online")
        return rewards

    return agent2_reward


# ─────────────────────────────────────────────────────────────────
# Training stages
# ─────────────────────────────────────────────────────────────────

def _grpo_args(output_dir: str) -> GRPOConfig:
    return GRPOConfig(
        output_dir                  = output_dir,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        num_train_epochs            = EPOCHS,
        learning_rate               = LR,
        warmup_ratio                = 0.05,
        lr_scheduler_type           = "cosine",
        num_generations             = NUM_GENERATIONS,
        max_new_tokens              = 512,
        logging_steps               = 5,
        save_steps                  = 50,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        report_to                   = "none",
    )


def stage1_train_agent1(model, tokenizer, use_online_reward: bool = True):
    print("\n" + "═" * 55)
    print("  Stage 1 — Training DiagnosticAgent (Agent 1)")
    print(f"  Reward range: [0, 1]  (normalised, max_mult={MAX_MULTIPLIER})")
    print("═" * 55)
    model   = apply_lora(model)
    dataset = build_agent1_dataset()
    trainer = GRPOTrainer(
        model         = model,
        tokenizer     = tokenizer,
        reward_funcs  = [make_a1_reward_online(ENV_URL) if use_online_reward else make_a1_reward()],
        args          = _grpo_args(str(CKPT_DIR / "agent1")),
        train_dataset = dataset,
    )
    trainer.train()

    # Save plots and CSV
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_rewards(trainer_a1=trainer, out_path=str(PLOTS_DIR / "agent1_reward_curve.png"))
    dump_trainer_rewards_to_csv(trainer, PLOTS_DIR / "agent1_trainer_rewards.csv", "agent1")

    model.save_pretrained(str(CKPT_DIR / "agent1" / "final"))
    tokenizer.save_pretrained(str(CKPT_DIR / "agent1" / "final"))
    print(f"✓ Agent 1 saved → {CKPT_DIR}/agent1/final")
    return model, trainer


def stage2_train_agent2(model, tokenizer, a1_model_fn, trainer_a1=None, use_online_reward: bool = True):
    print("\n" + "═" * 55)
    print("  Stage 2 — Training PatchAgent (Agent 2)")
    print(f"  Reward range: [0, 1]  (normalised, max_mult={MAX_MULTIPLIER})")
    print("═" * 55)
    model   = apply_lora(model)
    dataset = build_agent2_dataset(a1_model_fn)
    trainer = GRPOTrainer(
        model         = model,
        tokenizer     = tokenizer,
        reward_funcs  = [make_a2_reward_online(ENV_URL) if use_online_reward else make_a2_reward()],
        args          = _grpo_args(str(CKPT_DIR / "agent2")),
        train_dataset = dataset,
    )
    trainer.train()

    # Save plots, CSV, and model
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if trainer_a1 is not None:
        plot_rewards(
            trainer_a1 = trainer_a1,
            trainer_a2 = trainer,
            out_path   = str(PLOTS_DIR / "both_agents_reward_curves.png"),
        )
    plot_rewards(trainer_a2=trainer, out_path=str(PLOTS_DIR / "agent2_reward_curve.png"))
    dump_trainer_rewards_to_csv(trainer, PLOTS_DIR / "agent2_trainer_rewards.csv", "agent2")

    model.save_pretrained(str(CKPT_DIR / "agent2" / "final"))
    tokenizer.save_pretrained(str(CKPT_DIR / "agent2" / "final"))
    print(f"✓ Agent 2 saved → {CKPT_DIR}/agent2/final")
    return model, trainer


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate(a1_fn, a2_fn) -> dict:
    """
    Run all 3 tasks and print a summary table.
    Returns a dict of {task_id: result}.
    All rewards are ∈ [0, 1].
    """
    print("\n" + "─" * 65)
    print("EVALUATION — all 3 tasks   (rewards ∈ [0, 1])")
    print("─" * 65)
    a1   = DiagnosticAgent(a1_fn)
    a2   = PatchAgent(a2_fn)
    orch = RAGDiagnosticOrchestrator(a1, a2, base_url=ENV_URL)

    results = {}
    for tid in TASKS:
        result = asyncio.run(orch.run_episode(tid))
        results[tid] = result
        r = result["episode_total_reward"]
        gt_rc = TASKS[tid]["root_cause"]
        match_sym = "✓" if result["agent1"]["root_cause"] == gt_rc else "✗"
        print(f"\n[{tid}]  ({TASKS[tid]['difficulty']})")
        print(f"  Root cause  : {result['agent1']['root_cause']} {match_sym}  (GT: {gt_rc})")
        print(f"  Confidence  : {result['agent1']['confidence']:.2f}")
        print(f"  Patch       : {result['agent2']['patch']}")
        print(f"  GT patch    : {TASKS[tid]['correct_patch']}")
        print(f"  Episode reward : {r:.4f} / 1.0")
    print("\n" + "─" * 65)
    avg = sum(r["episode_total_reward"] for r in results.values()) / len(results)
    print(f"  Average episode reward: {avg:.4f} / 1.0")
    return results


def evaluate_random_baseline(trials_per_task: int = 5) -> dict:
    """Random baseline evaluated against the live environment."""
    known_root_causes = list({t["root_cause"] for t in TASKS.values()})

    async def _random_episode(task_id: str) -> float:
        async with RAGDiagnosticClient(ENV_URL) as client:
            await client.reset(task_id)
            obs = await client.step(DiagnoseAction(
                root_cause=random.choice(known_root_causes),
                explanation="Random guess.",
                confidence=0.5,
            ))
            obs = await client.step(PatchAction(patch={}, rationale="No patch."))
            return float(obs.info.get("episode_total_reward", 0.0))

    baseline: dict[str, float] = {}
    for tid in TASKS:
        scores = [asyncio.run(_random_episode(tid)) for _ in range(trials_per_task)]
        baseline[tid] = float(sum(scores) / len(scores))
    return baseline


def plot_baseline_vs_trained(baseline: dict, trained: dict, out_path: Path) -> None:
    task_labels = list(TASKS.keys())
    base_vals = [baseline.get(t, 0.0) for t in task_labels]
    train_vals = [trained.get(t, {}).get("episode_total_reward", 0.0) for t in task_labels]

    x = np.arange(len(task_labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, base_vals, width, label="Random baseline", color="#B0B3C6")
    ax.bar(x + width / 2, train_vals, width, label="Trained agent", color="#4F46E5")
    ax.set_xlabel("Task", fontsize=11)
    ax.set_ylabel("Episode reward (0-1)", fontsize=11)
    ax.set_title("Baseline vs trained reward by task", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n({TASKS[t]['difficulty']})" for t in task_labels], fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], help="Training stage")
    parser.add_argument("--eval",  action="store_true",      help="Run evaluation only")
    parser.add_argument(
        "--offline-reward",
        action="store_true",
        help="Use static/offline rewards instead of live environment rewards during training.",
    )
    args = parser.parse_args()
    use_online_reward = not args.offline_reward

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_base_model()

    trainer_a1 = None
    if args.stage == 1 or (not args.stage and not args.eval):
        model, trainer_a1 = stage1_train_agent1(
            model,
            tokenizer,
            use_online_reward=use_online_reward,
        )

    a1_fn = model_fn_from_weights(model, tokenizer)

    if args.stage == 2:
        model, _ = stage2_train_agent2(
            model,
            tokenizer,
            a1_fn,
            trainer_a1=trainer_a1,
            use_online_reward=use_online_reward,
        )

    a2_fn = model_fn_from_weights(model, tokenizer)

    if args.eval:
        trained = evaluate(a1_fn, a2_fn)
        baseline = evaluate_random_baseline()
        plot_path = PLOTS_DIR / "baseline_vs_trained.png"
        plot_baseline_vs_trained(baseline, trained, plot_path)
        summary = {
            "baseline": baseline,
            "trained": {k: v["episode_total_reward"] for k, v in trained.items()},
            "avg_baseline": sum(baseline.values()) / len(baseline),
            "avg_trained": sum(v["episode_total_reward"] for v in trained.values()) / len(trained),
            "mode": "online_env_reward" if use_online_reward else "offline_reward",
            "plot": str(plot_path),
        }
        (PLOTS_DIR / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"✓ Saved baseline comparison plot → {plot_path}")
        print(f"✓ Saved eval summary → {PLOTS_DIR / 'eval_summary.json'}")
