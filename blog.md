# Teaching 2 Agents to Debug RAG Pipelines

RAG pipelines break. A lot. And when they do, the failure modes are annoyingly specific: Chunk sizes bloat to 4096 tokens and the retriever starts returning garbage. The symptoms show up as low faithfulness scores and bad cosine similarities, but figuring out *why* — that's still a human job.

Agent 1 Diagnose the root cause and Agent 2 will fix it
This consists of 2 Themes in one project (Multi-Agent Interaction & World Modeling / Professional Tasks )

## Why two agents?

Early on I tried a single-agent approach: one model reads symptoms and outputs both the diagnosis and the patch. It sort of worked, but the reward signal was muddy. The model would sometimes get the diagnosis wrong but stumble into a correct patch through memorization. Or it'd nail the diagnosis but botch the patch format. Hard to tell what it was actually learning.

Splitting it into two agents cleaned things up a lot

- **Agent 1 (DiagnosticAgent)** only sees the symptoms. It outputs `{root_cause, explanation, confidence}`. It gets rewarded purely on diagnostic accuracy — did it identify the right failure mode?
- **Agent 2 (PatchAgent)** sees the symptoms *plus* Agent 1's diagnosis. It outputs a config patch. It gets rewarded on fix correctness and downstream faithfulness improvement.

This separation has a nice property: Agent 1 can't inflate its reward by guessing the patch, and Agent 2 can't mask a bad diagnosis. Each agent has to earn its own score. They share the same base model (Qwen2.5-7B with 4-bit quantization) but get separate LoRA adapters.

## The problem with debugging RAG

If you've operated a RAG system in production, you know the debugging loop. Something breaks. You stare at metrics — retrieval precision is 0.31, faithfulness is 0.44. You check the chunk config. You check the embedding dimensions. You grep through logs. Eventually you find it: `chunk_size=4096` was set by someone who thought bigger chunks meant better context. Classic.

The thing is, this kind of reasoning follows a pattern. There's a finite set of common failure modes. Each one produces a recognizable constellation of symptoms. An experienced engineer can usually narrow it down in a few minutes. So why not train a model to do the same?

That's what this project does. It's an RL environment — built on the OpenEnv framework — where an LLM learns to:

1. Look at pipeline symptoms (metrics, logs, configs)
2. Diagnose the root cause
3. Emit a config patch that fixes it


## The environment

The gym has three tasks, each representing a different RAG failure:

**Easy — Chunking gone wrong.** `chunk_size=4096` is way too big. Retrieval precision drops to 0.31 because the relevant answer is buried inside enormous text blocks. The fix is simple: set `chunk_size: 512, chunk_overlap: 64`.

**Medium — Embedding model mismatch.** The index was built with `text-embedding-ada-002` (1536-dim) but queries are encoded with `e5-large-v2` (1024-dim). The vectors live in completely different spaces. Cosine similarity averages 0.21 — basically random. Fix: switch the query encoder to match the index.

**Hard — Semantic drift without a reranker.** This one's tricky. Query expansion balloons queries from 8 to 47 tokens, causing semantic drift. Without a cross-encoder reranker to filter, drifted documents rank first. The misleading part: `retrieval_precision@3` is 0.52 — looks okayish — but `faithfulness_score` is 0.11. The model needs to look past the "acceptable" precision number and realize the retrieved docs are semantically wrong. The fix requires three config changes: disable query expansion, add a reranker, set top-k.

Each task comes with known distractor patches — plausible-looking fixes that don't actually help. The reward penalizes including these.

## Reward design

This took more iteration than the model architecture, honestly. The reward has three components:

```
R = difficulty_mult × (0.35 × R_diag + 0.45 × R_patch + 0.20 × R_faith)
```

Normalized so the final score is always in [0, 1].

**R_diag** grades the diagnosis. Exact root-cause match gets 1.0, partial keyword match gets 0.5, miss gets 0.0. There's also a small bonus for explanation quality (keyword overlap with the ground truth) and a calibration component. This last part matters: if the model says "confidence: 0.95" but the diagnosis is wrong, it takes a -0.15 penalty. If it's wrong but says "confidence: 0.3" — it gets a small bonus for knowing it's uncertain. The idea is to train epistemic humility, not just accuracy.

**R_patch** is F1 over config key-value pairs vs ground truth. If you include a distractor key, that's -0.08 per key. Perfect exact match gets a +0.05 bonus.

**R_faith** simulates the downstream faithfulness improvement. It interpolates between the baseline faithfulness and the target faithfulness proportionally to patch recall. If you fix all the right keys, you get the full improvement. If you only fix half, you get half.

Difficulty multipliers shape the curriculum: easy ×1.0, medium ×1.3, hard ×1.6. The dataset also oversamples harder tasks (easy gets 20 copies, medium 35, hard 50) so the model sees proportionally more gradient signal from challenging scenarios.

## Training

The whole thing runs on a free Colab T4 GPU, which was a hard constraint I imposed on myself. If it doesn't train on consumer hardware, it doesn't really prove the approach works.

**Stack:**
- Qwen2.5-7B-Instruct, 4-bit quantized via Unsloth
- LoRA rank 16 on all attention + MLP projections
- TRL's GRPOTrainer (Group Relative Policy Optimization)
- 8 completions per prompt for group-wise advantage estimation

The training loop has two stages. Stage 1 trains Agent 1's LoRA on diagnostic prompts. Stage 2 freezes Agent 1, uses it to generate diagnoses for each task, then trains Agent 2's LoRA on patch prompts. Both stages use online rewards — the completions are scored by the live environment server running in the background.

One thing that burned time: TRL changed their reward logging keys between versions. They used to log `rewards/mean`, now it's `rewards/{func_name}/mean` where `func_name` is the `__name__` attribute of your reward function. I was returning anonymous closures named `_fn`, so the logs were silently writing to keys nobody was reading. Took a while to figure out why my reward CSV was empty.

## What I'd do differently

**More tasks.** Three tasks is enough to show the approach works, but the model would generalize better with 10-15 failure modes. Things like: wrong top-k settings, missing metadata filters, outdated indices, prompt template errors. Each one follows the same pattern (symptoms → diagnosis → patch) so adding them is mostly data work.

**Better faithfulness simulation.** Right now R_faith is a linear interpolation based on patch recall. A more realistic version would actually simulate the post-patch retrieval with a real embedding model and corpus. I kept it simple for the hackathon timeline.

**Adversarial task generation.** Have another model generate *new* failure scenarios and reward functions, then test whether the diagnostic agent can generalize to unseen failure modes. That would be a much stronger test of whether it's actually learning causal reasoning vs pattern matching.

## Running it

The whole environment ships as a Docker container. The HuggingFace Space runs the FastAPI server with a Gradio UI on top. You can try it manually — pick a task, type in a diagnosis and patch, see how the reward breaks down.

For training, the Colab notebook handles everything: installs dependencies, starts the environment server in the background, runs both training stages, and generates comparison plots at the end.

The server exposes a WebSocket endpoint at `/ws` for the training loop, plus REST endpoints at `/health`, `/tasks`, and `/rubrics` for inspection. The rubric system is composable — each reward component is a separate class that can be scored independently.

---

**Links:**
- [HuggingFace Space](https://huggingface.co/spaces/szyyne/RAG_DIagnosis_v2)
- [Training Notebook (Colab)](https://colab.research.google.com/#fileId=https%3A//huggingface.co/spaces/szyyne/RAG_DIagnosis_v2/blob/main/rag_diagnosis_engine.ipynb)
- [GitHub](https://github.com/Szyyne/RAG_DIagnosis_v2)
