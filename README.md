# Recommendation Systems — Overview

A self-contained study archive for **industrial recommendation systems**, written for engineers preparing for ML / recommender-system interviews and for anyone who wants a path from first principles to the 2026 research frontier.

## What this archive teaches

The goal is to take you from the **mechanics of a production recommender** to the **research frontier**, and to make the connection between the two explicit at every step:

1. **The pipeline and its building blocks.** How a modern recommender turns a corpus of hundreds of millions of items into a short ranked list for one user — retrieval → pre-rank → rank → re-ranking — and the classic models at each stage (collaborative filtering, two-tower, MMoE, DCN, DIN/SIM, MMR/DPP), the metrics that drive them, and how changes are validated with A/B testing.
2. **How to actually move the numbers.** A practical playbook for improving the core metrics, plus the cold-start problem that every real system faces.
3. **Where the field is going.** The 2026 wave — generative recommendation and semantic IDs, scaling laws and unified Transformers, LLM-augmented and agentic recommenders, and the serving constraints that make or break all of it.

Two devices keep the theory grounded in practice:

- **Expanded Reading** — most fundamentals notes end with per-paper write-ups (method / results / trade-offs) of recent (2024–2026) work tied to that chapter's classic methods.
- **Pinterest in Practice** — many notes include a callout grounding the chapter in Pinterest's deployed systems, since this archive doubles as Pinterest interview prep. A standalone deep-dive lives in [Pinterest_RecSys_Recent_Progress.md](../../../Pinterest%20Recent%20Papers/Pinterest_RecSys_Recent_Progress.md).

## Table of contents

- [Fundamentals (01–08)](#fundamentals)
- [Frontier paradigms (09–12)](#frontier-paradigms)
- [Conventions](#conventions)
- [Adding new papers](#adding-new-papers)
- [Acknowledgement](#acknowledgement)

## Fundamentals

The classic production pipeline, using Xiaohongshu (小红书) as the running example.

| # | Note | Covers |
|---|------|--------|
| 01 | [Foundations & A/B Testing](01_Foundations_and_AB_Testing.md) | The pipeline (retrieval → pre-rank → rank → re-ranking), metrics (DAU / retention / LT), layered A/B testing, holdout |
| 02 | [Candidate Retrieval](02_Candidate_Retrieval.md) | ItemCF, Swing, UserCF, embeddings, matrix completion + ANN, two-tower, Deep Retrieval, Bloom filters |
| 03 | [Ranking Models](03_Ranking_Models.md) | Multi-task models, calibration, MMoE, score fusion, play-time modeling, features, pre-rank |
| 04 | [Feature Interaction](04_Feature_Interaction.md) | FM, DCN / DCNv2, LHUC / PPNet, SENet + Bilinear (FiBiNet) |
| 05 | [Sequence Modeling](05_Sequence_Modeling.md) | LastN, DIN (attention), SIM (long-sequence GSU/ESU search) |
| 06 | [Re-ranking & Diversity](06_Reranking_and_Diversity.md) | Item similarity, MMR, business rules, DPP |
| 07 | [Item Cold-Start](07_Item_Cold_Start.md) | Goals / metrics, simple channels, clustering retrieval, Look-Alike, flow control, cold-start A/B |
| 08 | [Metric Improvement Playbook](08_Metric_Improvement_Playbook.md) | Practical "how to move the metrics" playbook across retrieval / ranking / diversity + a 2026 frontier overlay |

## Frontier paradigms

Conceptual syntheses of the 2026 research wave. These are the homes for future papers — they explain *paradigms* holistically and cross-link to the per-paper detail in 01–08.

| # | Note | Covers |
|---|------|--------|
| 09 | [Generative Recommendation & Semantic IDs](09_Generative_Recommendation_and_Semantic_IDs.md) | Item-ID → semantic-ID shift, RQ-VAE / RQ-Kmeans / FSQ tokenization, generative retrieval, GR generalization theory, end-to-end GR ranking |
| 10 | [Scaling Laws & Large-Model Architectures](10_Scaling_Laws_and_Large_Model_Architectures.md) | Scaling laws for recsys, the unified feature-interaction + sequence Transformer wave (`*Former` family, HSTU lineage), lifelong sequences |
| 11 | [LLM-Augmented & Agentic Recommenders](11_LLM_Augmented_and_Agentic_Recommenders.md) | LLM as encoder / reasoner / agent: reasoning re-rankers, simulated environments, self-evolving AutoML, user & multimodal foundation models |
| 12 | [Serving & Inference Efficiency](12_Serving_and_Inference_Efficiency.md) | Latency budgets, constrained decoding on accelerators, KV-cache / RLB / sliding-window attention, MoE load balancing, custom GPU kernels (GDPA), MFU |

## Conventions

- **Citations** use APA. arXiv papers link as `https://arxiv.org/abs/<id>`; company posts cite the engineering blog.
- **Chinese terms** are given once, in parentheses on first mention (e.g., re-ranking (重排)), then referred to in English thereafter.
- **Math** is written in LaTeX. Each note opens with a table of contents and closes with a **Key Insights** section.

## Adding new papers

1. Decide whether the paper extends a **classic method** (→ add a per-paper entry to the relevant 01–08 *Expanded Reading* section) or advances a **frontier paradigm** (→ extend 09–12).
2. Most frontier papers fit both: write the depth in the frontier note, add a one-line pointer + APA citation in the related fundamentals note.
3. For a deployed industrial system, also add or extend a **Pinterest in Practice** (or analogous company) callout where relevant.
4. Keep citations APA; cross-link with relative markdown links (e.g. `[Note 10 …](10_Scaling_Laws_and_Large_Model_Architectures.md)`).

## Acknowledgement

The fundamentals (notes 01–08) are based on **Shusen Wang's (王树森)** publicly available *Recommendation System* course, with additions, the frontier notes, and all paper write-ups added by me.
