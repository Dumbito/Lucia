# Neurocognitive Design Notes — v0.1

This document records neuroscience-inspired mechanisms that may be translated into computational components of Lucía.

## Memory

### Working memory
A bounded active representation used by the reasoning model. It should be treated as a computational workspace rather than permanent storage.

### Episodic memory
Stores significant experiences/events with temporal and contextual metadata.

### Semantic memory
Stores consolidated concepts and stable knowledge derived from multiple experiences.

### Procedural memory
Stores reusable procedures, strategies and action patterns.

### Consolidation
Repeated, relevant or strongly informative experiences can be transformed from transient episodes into more stable knowledge.

### Reconsolidation
Retrieved memories may be updated when new evidence modifies their interpretation.

### Forgetting and decay
Memories that are rarely useful can lose retrieval priority over time without necessarily being immediately deleted.

### Interference
Similar or contradictory memories can compete during retrieval. The memory system should represent uncertainty and evidence rather than blindly selecting one record.

### Pattern separation and completion
The system may distinguish similar episodes while still allowing partial cues to recover a related experience.

## Learning

### Plasticity-inspired updates
Associations between concepts, contexts, procedures and outcomes may gain or lose strength based on experience.

### Reinforcement / prediction error
Action strategies can be evaluated against expected outcomes. Positive or negative prediction errors can influence future strategy selection.

### Active learning
When uncertainty and expected information value are high, Lucía may seek additional information or request confirmation instead of consolidating an uncertain inference.

### Metacognition
Memories, hypotheses and conclusions should carry confidence, evidence and contradiction metadata.

## Attention and salience

Lucía should not send every environmental event to an expensive model. Candidate events can be ranked using factors such as novelty, relevance to active goals, urgency, uncertainty and interruption cost.

A conceptual salience score can be represented as:

`S = f(novelty, relevance, urgency, uncertainty, goal_alignment, interruption_cost)`

The exact function will be determined experimentally.

## Offline consolidation

An idle/offline process may periodically review recent experiences, merge redundant memories, detect contradictions, update associations, and promote useful knowledge. This is an engineering analogy to memory consolidation, not a claim that the process reproduces biological sleep.

## Research discipline

Every neuro-inspired mechanism should eventually have:

1. a clearly defined computational hypothesis;
2. an implementation;
3. measurable metrics;
4. an ablation or baseline comparison;
5. documented limitations.
