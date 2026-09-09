# Lucía Architecture — v0.1

## 1. System definition

Lucía is a persistent personal AI system. The language model is one component of the system, not the system itself.

## 2. High-level flow

```text
Environment
    ↓
Perception
    ↓
Salience / Attention
    ↓
Context Engine ← Memory Retrieval
    ↓
Cognition / Planning
    ↓
Model Router
    ↓
Tools / Actions
    ↓
Outcome Evaluation
    ↓
Memory + Learning
    ↓
Updated Context
```

## 3. Major modules

- `core`: lifecycle, identity and system coordination.
- `context`: current state, active goals, recent events and assembled model context.
- `memory`: persistent episodic, semantic, procedural, project and preference memories.
- `learning`: consolidation, reinforcement, forgetting, confidence and experience-driven updates.
- `perception`: computer/environment observations and event extraction.
- `cognition`: reasoning, planning and evaluation interfaces.
- `agents`: controlled multi-step task execution.
- `tools`: filesystem, terminal, Git, web and other explicitly authorized capabilities.
- `voice`: speech input/output interfaces.
- `avatar`: visual presence and state expression.

## 4. Model strategy

Local models are treated as interchangeable engines selected by task requirements. Initial candidates:

- GPT-OSS 20B — reasoning and long-context work.
- Qwen3-Coder 30B-A3B — complex coding and agentic coding.
- Qwen2.5-Coder 14B — fast coding.
- Qwen3 14B — general-purpose work.
- Qwen3 8B — low-latency/simple tasks.

The exact runtime, quantization and routing policy will be benchmarked before being frozen.

## 5. Context vs memory

**Context** is working state: what is happening now and what the selected model needs for the current inference.

**Memory** is persistent information outside the model's context window. A retrieval/consolidation layer determines what becomes available to the current context.

## 6. Neuro-inspired direction

Potential mechanisms include:

- working-memory limits
- episodic/semantic/procedural separation
- temporal decay
- retrieval-based strengthening
- consolidation
- reconsolidation
- interference
- pattern separation/completion
- salience and attention
- reinforcement and prediction error
- confidence/metacognition
- offline consolidation

These mechanisms must be implemented as measurable computational hypotheses rather than presented as literal simulations of the brain.

## 7. Human-like presence

Warmth and continuity should emerge from persistent context, selective memory, curiosity, appropriate initiative, voice, avatar state and sensitivity to the user's activity. Lucía should not continuously interrupt the user; initiative is governed by relevance, novelty, urgency, uncertainty and interruption cost.

## 8. Development principle

Build the cognitive/data architecture first. Add voice, perception and avatar after the core loop is stable. Avoid unnecessary dependencies until their role is justified by an architectural requirement.
