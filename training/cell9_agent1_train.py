# ── 9. Train Agent 1 (DiagnosticAgent) — FIXED ───────────────────────
# DEMO_MODE=True → quick 10-step pipeline validation
# DEMO_MODE=False → full training (~40 min on T4)
DEMO_MODE = True   # ← set False for full training

import os, csv, json, re, asyncio, torch
from datetime import datetime
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from training.train_grpo import (
    make_a1_reward, make_a1_reward_online,
    model_fn_from_weights, dump_trainer_rewards_to_csv,
)
from rag_diagnostic_gym.client import RAGDiagnosticClient
from rag_diagnostic_gym.models import DiagnoseAction, Observation
from rag_diagnostic_gym.tasks import TASKS
from agents.orchestrator import diagnostic_prompt   # ← builds proper string prompt

os.makedirs("/content/checkpoints/agent1", exist_ok=True)
os.makedirs("/content/plots",              exist_ok=True)
os.makedirs("/content/logs",               exist_ok=True)
if DEMO_MODE:
    os.environ["WANDB_DISABLED"] = "true"

EPOCHS_DEMO, STEPS_DEMO = 1, 10
NUM_EPOCHS = EPOCHS_DEMO if DEMO_MODE else 3

# True  = reward calls live environment (online RL)
# False = reward computed from static labels/rubrics (offline RL)
USE_ONLINE_ENV_REWARD = True

# ── CSV logger ────────────────────────────────────────────────────────
LOG_PATH_A1 = "/content/logs/agent1_rewards.csv"
with open(LOG_PATH_A1, "w", newline="") as f:
    csv.writer(f).writerow(["timestamp", "epoch", "step",
                             "task_id", "difficulty", "reward", "source"])

def _log_a1(epoch, step, task_id, reward, source):
    diff = TASKS[task_id]["difficulty"] if task_id in TASKS else "internal"
    with open(LOG_PATH_A1, "a", newline="") as f:
        csv.writer(f).writerow([datetime.utcnow().isoformat(),
                                 epoch, step, task_id, diff,
                                 round(float(reward), 6), source])

# ── Parse raw LLM text → dict ────────────────────────────────────────
def _parse_json(text: str) -> dict:
    """Robust JSON extractor from model output."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}

# ── Live env episode for Agent 1 ──────────────────────────────────────
async def _a1_env_eval(a1_fn, epoch):
    # FIX 1: use localhost (0.0.0.0 is for binding, not connecting)
    async with RAGDiagnosticClient("ws://localhost:8000/ws") as client:
        for i, task_id in enumerate(TASKS):
            obs = await client.reset(task_id)

            # FIX 2: build a proper string prompt (a1_fn expects str, not dict)
            prompt = diagnostic_prompt(obs)

            # FIX 3: a1_fn returns a STRING, not a dict — must parse JSON
            raw_text = a1_fn(prompt)
            parsed   = _parse_json(raw_text)

            obs = await client.step(DiagnoseAction(
                root_cause  = parsed.get("root_cause", "unknown"),
                explanation = parsed.get("explanation", ""),
                confidence  = float(parsed.get("confidence", 0.5)),
            ))
            _log_a1(epoch, epoch * len(TASKS) + i, task_id, obs.reward, "env")
            print(f"  [A1 env | epoch={epoch} | {task_id}] reward={obs.reward:.4f}"
                  f" | {obs.reward_breakdown}")

# ── Helper: find reward key in trainer log_history ────────────────────
def _find_reward_key(log_history):
    """TRL logs rewards under different keys depending on version."""
    for e in log_history:
        for k in e:
            # New TRL: rewards/agent1_reward/mean
            if k.startswith("rewards/") and k.endswith("/mean"):
                return k
        # Old TRL
        if "rewards/mean" in e:
            return "rewards/mean"
        if "reward/mean" in e:
            return "reward/mean"
    return None

# ── Build dataset ─────────────────────────────────────────────────────
ds_a1_train = Dataset.from_list(list(ds_a1)[:8]) if DEMO_MODE else ds_a1

# ── Pick reward function ──────────────────────────────────────────────
a1_reward_fn = (
    make_a1_reward_online("ws://localhost:8000/ws")
    if USE_ONLINE_ENV_REWARD
    else make_a1_reward()
)

trainer_a1 = GRPOTrainer(
    model         = model,
    tokenizer     = tokenizer,
    reward_funcs  = [a1_reward_fn],
    train_dataset = ds_a1_train,
    args = GRPOConfig(
        output_dir                  = "/content/checkpoints/agent1",
        per_device_train_batch_size = 2 if DEMO_MODE else 4,
        gradient_accumulation_steps = 2 if DEMO_MODE else 4,
        num_train_epochs            = 1,           # 1 epoch per loop iteration
        max_steps                   = STEPS_DEMO if DEMO_MODE else -1,
        learning_rate               = 2e-5,
        num_generations             = 4 if DEMO_MODE else 8,
        max_new_tokens              = 256,
        logging_steps               = 1,
        save_steps                  = 5 if DEMO_MODE else 50,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        report_to                   = "none",
    ),
)

# ── Training loop with per-epoch env eval + CSV logging ───────────────
mode_name = "ONLINE env reward" if USE_ONLINE_ENV_REWARD else "OFFLINE static reward"
print(f"Training Agent 1 ({mode_name}, rewards ∈ [0, 1])…")

for epoch in range(NUM_EPOCHS):
    print(f"\n── Epoch {epoch + 1}/{NUM_EPOCHS} ──")

    # Train one epoch
    trainer_a1.train(resume_from_checkpoint=True if epoch > 0 else None)

    # ── Log GRPO rewards from trainer history to CSV ──────────────
    rk = _find_reward_key(trainer_a1.state.log_history)
    if rk:
        for entry in trainer_a1.state.log_history:
            if rk in entry:
                _log_a1(epoch, entry.get("step", -1),
                        "grpo_internal", entry[rk], "grpo")
        last_r = [entry[rk] for entry in trainer_a1.state.log_history if rk in entry]
        if last_r:
            print(f"  GRPO reward key: {rk}")
            print(f"  Last logged reward: {last_r[-1]:.4f} / 1.0")
    else:
        print("  ⚠ No reward key found in log_history")
        avail = set(k for e in trainer_a1.state.log_history for k in e)
        print(f"  Available keys: {avail}")

    # ── Run live env evaluation ───────────────────────────────────
    print(f"\n  Running Agent 1 live env evaluation (epoch {epoch})…")
    a1_fn = model_fn_from_weights(model, tokenizer, temperature=0.3)
    try:
        # Handle Jupyter's already-running event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're in Jupyter — use nest_asyncio or thread
            import threading
            result = {}
            exc = {}
            def _run():
                try:
                    asyncio.run(_a1_env_eval(a1_fn, epoch))
                except Exception as e:
                    exc["err"] = e
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=120)
            if "err" in exc:
                raise exc["err"]
        else:
            asyncio.run(_a1_env_eval(a1_fn, epoch))
    except Exception as e:
        print(f"  ⚠ Env eval failed: {e}")

    # ── Save checkpoint ───────────────────────────────────────────
    ckpt = f"/content/checkpoints/agent1/epoch_{epoch + 1}"
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    print(f"  ✓ Checkpoint saved → {ckpt}")

# ── Post-training: dump trainer rewards to CSV as backup ──────────────
dump_trainer_rewards_to_csv(
    trainer_a1,
    "/content/logs/agent1_trainer_rewards.csv",
    "agent1",
)

# ── Final model save ─────────────────────────────────────────────────
model.save_pretrained("/content/checkpoints/agent1/final")
tokenizer.save_pretrained("/content/checkpoints/agent1/final")
print("✓ Agent 1 trained and saved → /content/checkpoints/agent1/final")

# ── Summary ──────────────────────────────────────────────────────────
import pandas as pd
try:
    df = pd.read_csv(LOG_PATH_A1)
    print(f"\n📊 Logged {len(df)} reward entries to {LOG_PATH_A1}")
    if not df.empty:
        print(df.groupby("source")["reward"].agg(["count", "mean", "max"]))
except Exception:
    pass
