"""
HuggingFace Spaces entry point — RAG Diagnostic Gym.

Serves two things in one process:
  1. Gradio UI  (port 7860, public demo)
  2. FastAPI/WS env server  (mounted at /env/*)

Visitors can interact with the environment directly in the browser,
and the WebSocket endpoint is accessible for programmatic clients.
"""
from __future__ import annotations

import json
import os
import threading
import traceback

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import uvicorn
from fastapi import FastAPI
from huggingface_hub import InferenceClient

from rag_diagnostic_gym.server.environment import create_app
from rag_diagnostic_gym.server.environment import RAGDiagnosticEnvironment
from rag_diagnostic_gym.models import DiagnoseAction, PatchAction
from rag_diagnostic_gym.tasks import TASKS
from agents.orchestrator import diagnostic_prompt, patch_prompt

# ── FastAPI env server runs in background thread ──────────────────
env_app: FastAPI = create_app()

def _run_env_server():
    uvicorn.run(env_app, host="0.0.0.0", port=8765, log_level="warning")

threading.Thread(target=_run_env_server, daemon=True).start()


# ── Per-session environment state ─────────────────────────────────
_envs: dict[str, RAGDiagnosticEnvironment] = {}

def _get_env(session_id: str) -> RAGDiagnosticEnvironment:
    if session_id not in _envs:
        _envs[session_id] = RAGDiagnosticEnvironment()
    return _envs[session_id]


# ── HuggingFace Inference helper ──────────────────────────────────

HF_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
]

def _parse_llm_json(text: str) -> dict:
    """Robustly parse JSON from LLM output (handles markdown fences, extra text)."""
    import re
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError(f"Could not parse JSON from LLM output:\n{text[:500]}")


def _call_hf_llm(prompt: str, hf_token: str, model_id: str) -> str:
    """Call HuggingFace Inference API with a prompt, return the text completion."""
    client = InferenceClient(token=hf_token)
    response = client.chat_completion(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a senior RAG pipeline engineer. Respond ONLY with valid JSON, no markdown fences, no extra text."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
        temperature=0.3,
    )
    return response.choices[0].message.content


# ── Gradio Callbacks ──────────────────────────────────────────────

def do_reset(task_choice: str, session_id: str) -> tuple[str, str, str, str, str]:
    env = _get_env(session_id)
    tid = None if task_choice == "Random" else task_choice
    try:
        obs = env.reset(task_id=tid)
    except Exception as e:
        return ("", f"❌ Reset error: {e}", "", "", "")
    return (
        json.dumps(obs.symptoms, indent=2),
        f"**Task:** `{obs.task_id}` | **Difficulty:** `{obs.difficulty}` | **Step:** {obs.step}/2",
        "",   # clear diagnosis box
        "",   # clear diag info
        "",   # clear result box
    )


def do_auto_solve(task_choice: str, model_id: str, session_id: str):
    """
    Auto-solve: reset the environment, call LLM for diagnosis, call LLM for patch.
    Uses a generator to stream progress updates to the UI.
    """
    env = _get_env(session_id)
    tid = None if task_choice == "Random" else task_choice

    # ── Reset ─────────────────────────────────────────────────────
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token or hf_token == "your_huggingface_token_here":
        yield ("", "❌ **HF_TOKEN not set.** Add your token to the `.env` file and restart.", "", "", "", "")
        return

    try:
        obs = env.reset(task_id=tid)
    except Exception as e:
        yield ("", f"❌ Reset error: {e}", "", "", "", "")
        return

    symptoms_json = json.dumps(obs.symptoms, indent=2)
    status = f"**Task:** `{obs.task_id}` | **Difficulty:** `{obs.difficulty}` | **Step:** 0/2"

    # Show symptoms, indicate Agent 1 is working
    yield (
        symptoms_json,
        status,
        "",
        "⏳ **Agent 1 (DiagnosticAgent)** is analyzing symptoms...",
        "",
        "",
    )

    # ── Agent 1: Diagnose ─────────────────────────────────────────
    try:
        prompt1 = diagnostic_prompt(obs)
        raw1 = _call_hf_llm(prompt1, hf_token, model_id)
        parsed1 = _parse_llm_json(raw1)

        action1 = DiagnoseAction(
            root_cause=parsed1.get("root_cause", "unknown"),
            explanation=parsed1.get("explanation", ""),
            confidence=float(parsed1.get("confidence", 0.5)),
        )
        obs = env.step(action1)
    except Exception as e:
        yield (
            symptoms_json, status, "",
            f"❌ **Agent 1 failed:** {e}\n\n**Raw LLM output:**\n```\n{raw1 if 'raw1' in dir() else 'N/A'}\n```",
            "", "",
        )
        return

    diag_json = json.dumps(obs.diagnosis, indent=2)
    diag_info = (
        f"✅ **Agent 1 complete!**\n\n"
        f"**Step reward:** `{obs.reward:.4f}`\n\n"
        f"**Breakdown:**\n```json\n{json.dumps(obs.reward_breakdown.model_dump() if obs.reward_breakdown else {}, indent=2)}\n```\n\n"
        f"**Ground truth root cause:** `{obs.info.get('ground_truth_root_cause', '?')}`"
    )

    # Show diagnosis, indicate Agent 2 is working
    yield (
        symptoms_json,
        f"**Task:** `{obs.task_id}` | **Difficulty:** `{obs.difficulty}` | **Step:** 1/2",
        diag_json,
        diag_info,
        "",
        "⏳ **Agent 2 (PatchAgent)** is generating config patch...",
    )

    # ── Agent 2: Patch ────────────────────────────────────────────
    try:
        prompt2 = patch_prompt(obs)
        raw2 = _call_hf_llm(prompt2, hf_token, model_id)
        parsed2 = _parse_llm_json(raw2)

        action2 = PatchAction(
            patch=parsed2.get("patch", {}),
            rationale=parsed2.get("rationale", ""),
        )
        obs = env.step(action2)
    except Exception as e:
        yield (
            symptoms_json,
            f"**Task:** `{obs.task_id}` | **Difficulty:** `{obs.difficulty}` | **Step:** 1/2",
            diag_json, diag_info,
            "",
            f"❌ **Agent 2 failed:** {e}\n\n**Raw LLM output:**\n```\n{raw2 if 'raw2' in dir() else 'N/A'}\n```",
        )
        return

    result = {
        "episode_total_reward": obs.info.get("episode_total_reward"),
        "submitted_patch": obs.info.get("submitted_patch"),
        "ground_truth_patch": obs.info.get("ground_truth_patch"),
        "patch_breakdown": obs.info.get("patch_breakdown"),
        "faithfulness_score": obs.info.get("faithfulness_score"),
    }
    total_reward = obs.info.get("episode_total_reward", 0)

    # Score emoji
    if total_reward >= 0.8:
        emoji = "🏆"
    elif total_reward >= 0.5:
        emoji = "✅"
    elif total_reward >= 0.3:
        emoji = "⚠️"
    else:
        emoji = "❌"

    patch_result = (
        f"{emoji} **Episode complete!**\n\n"
        f"### Total Reward: `{total_reward:.4f}`\n\n"
        f"**Submitted patch:**\n```json\n{json.dumps(action2.patch, indent=2)}\n```\n\n"
        f"**Ground truth patch:**\n```json\n{json.dumps(obs.info.get('ground_truth_patch', {}), indent=2)}\n```\n\n"
        f"**Full breakdown:**\n```json\n{json.dumps(result, indent=2)}\n```"
    )

    yield (
        symptoms_json,
        f"**Task:** `{obs.task_id}` | **Difficulty:** `{obs.difficulty}` | **Step:** 2/2 ✅",
        diag_json,
        diag_info,
        "",
        patch_result,
    )


def do_manual_diagnose(root_cause: str, explanation: str, confidence: float, session_id: str) -> tuple[str, str]:
    env = _get_env(session_id)
    if env.state is None:
        return "", "⚠️ Please click **Reset Episode** first."
    if env.state.step != 0:
        if env.state.diagnosis:
            return json.dumps(env.state.diagnosis, indent=2), "ℹ️ Diagnosis already submitted. Submit a **Patch** next, or **Reset**."
        return "", "⚠️ Unexpected state. Click **Reset Episode**."
    try:
        action = DiagnoseAction(root_cause=root_cause, explanation=explanation, confidence=confidence)
        obs = env.step(action)
        diag_display = json.dumps(obs.diagnosis, indent=2)
        info = (
            f"**Step reward:** `{obs.reward:.4f}`\n\n"
            f"**Breakdown:** {json.dumps(obs.reward_breakdown.model_dump() if obs.reward_breakdown else {}, indent=2)}\n\n"
            f"**Hint:** {obs.info.get('hint', '')}"
        )
        return diag_display, info
    except Exception as e:
        return "", f"❌ Error: {e}"


def do_manual_patch(patch_json: str, rationale: str, session_id: str) -> str:
    env = _get_env(session_id)
    if env.state is None:
        return "⚠️ Please click **Reset Episode** first."
    if env.state.step == 0:
        return "⚠️ Please submit a **Diagnosis** first (Step 1)."
    if env.state.terminated:
        return "ℹ️ Episode already complete. Click **Reset Episode** to start a new one."
    try:
        patch = json.loads(patch_json)
        action = PatchAction(patch=patch, rationale=rationale)
        obs = env.step(action)
        result = {
            "episode_total_reward": obs.info.get("episode_total_reward"),
            "ground_truth_patch": obs.info.get("ground_truth_patch"),
            "submitted_patch": obs.info.get("submitted_patch"),
            "patch_breakdown": obs.info.get("patch_breakdown"),
            "faithfulness_score": obs.info.get("faithfulness_score"),
        }
        return (
            f"✅ **Episode complete!**\n\n"
            f"**Total reward:** `{obs.info.get('episode_total_reward', 0):.4f}`\n\n"
            f"```json\n{json.dumps(result, indent=2)}\n```"
        )
    except Exception as e:
        return f"❌ Error: {e}"


# ── Gradio UI ──────────────────────────────────────────────────────

TASK_CHOICES = ["Random"] + list(TASKS.keys())

DESCRIPTION = """
## 🔬 RAG Diagnostic Gym
**OpenEnv v0.2.2 RL environment** for training LLMs to diagnose broken RAG pipelines.

**How it works:** Select a task → click **🤖 Auto-Solve** → watch two LLM agents automatically diagnose the root cause and generate a config patch.

| Task | Difficulty | Multiplier |
|------|-----------|-----------|
| `chunking_error_001` | Easy | ×1.0 |
| `embedding_mismatch_001` | Medium | ×1.3 |
| `hallucination_retrieval_001` | Hard | ×1.6 |

**Reward:** `R = multiplier × (0.35×R_diag + 0.45×R_patch + 0.20×R_faith)`
"""

with gr.Blocks(title="RAG Diagnostic Gym") as demo:
    session_id = gr.State(lambda: str(id({})))

    gr.Markdown(DESCRIPTION)

    # ── Configuration ──────────────────────────────────────────────
    with gr.Row():
        task_dd = gr.Dropdown(TASK_CHOICES, value="Random", label="🎯 Task")
        model_dd = gr.Dropdown(HF_MODELS, value=HF_MODELS[0], label="🧠 Model")

    # ── Auto-Solve Button ──────────────────────────────────────────
    with gr.Row():
        auto_btn = gr.Button("🤖 Auto-Solve (Reset + Diagnose + Patch)", variant="primary", size="lg")

    status_md = gr.Markdown("*Select a task and click Auto-Solve to begin.*")

    with gr.Row():
        symptoms_box = gr.Code(label="📊 Observable Symptoms", language="json", lines=15, interactive=False)

    # ── Agent 1 Results ────────────────────────────────────────────
    gr.Markdown("### 🔍 Agent 1 — Diagnosis")
    diag_out = gr.Code(label="Diagnosis Payload", language="json", lines=6, interactive=False)
    diag_info = gr.Markdown()

    # ── Agent 2 Results ────────────────────────────────────────────
    gr.Markdown("### 🛠 Agent 2 — Patch")
    agent2_status = gr.Markdown()
    patch_out = gr.Markdown()

    # ── Wire Auto-Solve ────────────────────────────────────────────
    auto_btn.click(
        do_auto_solve,
        inputs=[task_dd, model_dd, session_id],
        outputs=[symptoms_box, status_md, diag_out, diag_info, agent2_status, patch_out],
    )

    # ── Manual Mode (collapsible) ──────────────────────────────────
    with gr.Accordion("🔧 Manual Mode (type your own diagnosis & patch)", open=False):
        reset_btn = gr.Button("🔄 Reset Episode", variant="secondary")
        manual_diag_info = gr.Markdown()

        gr.Markdown("#### Step 1 — Diagnose")
        with gr.Row():
            rc_input = gr.Textbox(label="root_cause (snake_case)", placeholder="e.g. chunk_size_too_large")
            conf_input = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="confidence")
        expl_input = gr.Textbox(label="explanation", lines=3,
                                placeholder="Explain why this root cause produces the observed symptoms...")
        manual_diag_btn = gr.Button("🔍 Submit Diagnosis", variant="primary")
        manual_diag_out = gr.Code(label="Diagnosis Payload", language="json", lines=6, interactive=False)
        manual_diag_result = gr.Markdown()

        gr.Markdown("#### Step 2 — Patch")
        patch_input = gr.Code(label="patch (JSON)", language="json", lines=5,
                              value='{\n  "chunk_size": 512\n}')
        rationale_input = gr.Textbox(label="rationale", lines=2)
        manual_patch_btn = gr.Button("🛠 Submit Patch", variant="primary")
        manual_patch_out = gr.Markdown()

        # Wire manual callbacks
        reset_btn.click(do_reset,
                        inputs=[task_dd, session_id],
                        outputs=[symptoms_box, status_md, manual_diag_out, manual_diag_result, manual_patch_out])

        manual_diag_btn.click(do_manual_diagnose,
                              inputs=[rc_input, expl_input, conf_input, session_id],
                              outputs=[manual_diag_out, manual_diag_result])

        manual_patch_btn.click(do_manual_patch,
                               inputs=[patch_input, rationale_input, session_id],
                               outputs=[manual_patch_out])


if __name__ == "__main__":
    demo.launch(
    server_name="0.0.0.0",        # ← critical
    server_port=7860,             # ← optional but explicit
    share=False                   # share=True is ignored on Spaces
)
