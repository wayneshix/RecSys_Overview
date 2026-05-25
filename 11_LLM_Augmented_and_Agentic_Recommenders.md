# Part 11 — LLM-Augmented & Agentic Recommenders

> Recommendation systems course notes — Part 11: a conceptual / synthesis chapter on how Large Language Models (LLMs, 大语言模型) are entering the recommendation pipeline.
> This note is a **paradigm map**, not a per-paper deep-dive. It synthesizes the frontier and cross-references the per-paper "Expanded Reading" write-ups in Notes 05, 06, and 07.
> English with key Chinese terms in parentheses. ML-interview depth.

This chapter sits on top of the classical recsys pipeline (召回 → 粗排 → 精排 → 重排) that Notes 01–08 build up. Those notes optimize *deep-learning recommendation models* (DLRMs) with ID embeddings, two-tower retrieval, attention sequence models, and hand-crafted re-ranking objectives. Here we ask a different question: **where does an LLM actually help, and what does it cost?** The honest answer is that LLMs do not replace the pipeline — they slot into specific stages, each with a distinct mechanism, payoff, and serving-cost profile.

---

## Table of Contents

1. [A Framing Spectrum: Three Roles for LLMs in RecSys](#1-a-framing-spectrum-three-roles-for-llms-in-recsys)
   - 1.1 [LLM-as-encoder / foundation model](#11-llm-as-encoder--foundation-model-表征--基座模型)
   - 1.2 [LLM-as-reasoner / reranker](#12-llm-as-reasoner--reranker-推理器--重排)
   - 1.3 [LLM-as-agent](#13-llm-as-agent-智能体)
   - 1.4 [Where each role sits in the pipeline](#14-where-each-role-sits-in-the-pipeline)
2. [Reasoning Re-Rankers in Depth](#2-reasoning-re-rankers-in-depth)
   - 2.1 [From a fixed set objective to a learned reasoning policy](#21-from-a-fixed-set-objective-to-a-learned-reasoning-policy)
   - 2.2 [RL reward design for ranking a slate](#22-rl-reward-design-for-ranking-a-slate)
   - 2.3 [Reward hacking](#23-reward-hacking-奖励作弊)
   - 2.4 [Contrast with MMR / DPP (Note 06)](#24-contrast-with-mmr--dpp-note-06)
3. [Agentic RecSys: Simulated Environments & Self-Evolving Pipelines](#3-agentic-recsys-simulated-environments--self-evolving-pipelines)
   - 3.1 [Why offline metrics and A/B are insufficient](#31-why-offline-metrics-and-ab-are-insufficient)
   - 3.2 [What a world model / simulated user gives you](#32-what-a-world-model--simulated-user-gives-you)
   - 3.3 [Self-evolving pipelines: AutoML by LLM agents](#33-self-evolving-pipelines-automl-by-llm-agents)
4. [Multimodal & Reasoning-Aligned Representations](#4-multimodal--reasoning-aligned-representations)
5. [Trade-offs: When Is an LLM Worth It?](#5-trade-offs-when-is-an-llm-worth-it)
6. [Interview Cheat-Sheet](#6-key-insights)
7. [References](#7-references)

---

## 1. A Framing Spectrum: Three Roles for LLMs in RecSys

The cleanest way to organize the 2025–2026 literature is by **what job the LLM does**, ordered from cheapest/most-mature to most-experimental:

| Role | LLM does… | Output consumed by | Online cost | Maturity |
|---|---|---|---|---|
| **(a) Encoder / FM** | produce a representation (user / item / content embedding) | downstream DLRM | precompute offline; cheap at serving | production |
| **(b) Reasoner / reranker** | read context, reason, emit an ordering | the user directly (final slate) | autoregressive decode per request — heavy | early production / research |
| **(c) Agent** | author objectives, simulate users, rewrite the pipeline | the *training/dev loop*, not the request | offline meta-loop — slow but off the request path | research / first deployments |

The three roles differ in **where the LLM's cost lands**: (a) pushes it offline as a precomputed feature, (b) puts it directly on the serving request path, and (c) moves it into the development/training loop. That distinction drives the whole trade-off discussion in §5.

### 1.1 LLM-as-encoder / foundation model (表征 / 基座模型)

The most mature and lowest-risk role. The LLM (or a Transformer trained LLM-style) is a **frozen or periodically-retrained encoder** that turns raw signal into a dense vector; that vector is then a *feature* for the ordinary DLRM. Crucially, the expensive forward pass runs **offline / batch**, the embedding is cached, and serving stays cheap.

Two flavors already covered in earlier notes:

- **User foundation models (用户基座模型).** Pre-train one reusable user vector from a long action sequence, self-supervised, task-agnostic — then reuse it across many downstream tasks. See **Coinbase's user foundation model** in [Note 05 — Sequence Modeling](05_Sequence_Modeling.md): a two-tower Transformer over months of interactions, trained by same-user/different-user sequence-pair classification, consumed as a similarity index, a static feature, or a fine-tuned block. This is the "LastN-as-foundation" idea — encode the sequence *once* instead of averaging it into a task-specific feature.
- **Multimodal content foundation models (多模态内容基座模型).** Embed an item from its raw content (image/video/audio/text), giving a usable vector at launch *before* any interaction — the cold-start signal. See **Netflix MediaFM** in [Note 07 — Item Cold-Start](07_Item_Cold_Start.md): a tri-modal self-supervised Transformer whose embeddings feed clustering-retrieval and content matching, the production-scale descendant of the CNN+BERT content model in Note 06 §2.3.

These are "LLM-augmented" rather than "LLM-as-recommender": the LLM never sees the request; it manufactures better features the cheap model consumes.

### 1.2 LLM-as-reasoner / reranker (推理器 / 重排)

Here the LLM is **on the request path** and produces the actual recommendation decision — typically the final *ordering of a small candidate slate*. The mechanism is chain-of-thought (CoT, 思维链) reasoning over item semantics + user history, tuned with reinforcement learning. The flagship example is **GR2 (Generative Reasoning Re-Ranker)** from [Note 06 — Re-ranking & Diversity](06_Reranking_and_Diversity.md)'s Expanded Reading. Its mechanism, in one line: **Semantic-ID mid-training → reasoning SFT → RL (DAPO)**. Expanded in §2.

### 1.3 LLM-as-agent (智能体)

The LLM is an **autonomous agent** that acts over multiple steps with perception, memory, planning, and tool use — but the agent operates in the **development/training loop**, not (yet) the per-request serving path. Two complementary instances, both cross-referenced from [Note 06](06_Reranking_and_Diversity.md):

- **RecoWorld** — a *simulated user environment*: an LLM "user" and an agentic recommender interact over multiple turns, the user issuing natural-language instructions ("too many hairstyling clips, show something different but related"), enabling **multi-turn RL** against a retention signal (§3.1–§3.2).
- **Self-Evolving Recommendation System** — LLM agents act as *ML engineers*, authoring new architectures, optimizers, and reward functions and deploying them through inner/outer AutoML loops (§3.3).

### 1.4 Where each role sits in the pipeline

```
                 ┌─────────────── role (a): LLM-as-encoder ───────────────┐
                 │  user FM (Note 05) · content FM / MediaFM (Note 07)     │
                 │  QARM V2 multimodal SIDs (Note 05)                      │
                 ▼  (offline-precomputed embeddings, fed as features)      ▼
  recall → pre-rank → rank → re-rank → slate 1..k
                                                       ▲
                                                       │ role (b): LLM-as-reasoner
                                                       │ GR2 reasoning re-ranker (Note 06)
                                                       │ (on the request path, small n)

  role (c): LLM-as-agent — OFF the request path, in the dev/training loop:
    RecoWorld (simulated user → multi-turn RL)   ·   Self-Evolving (agents author & deploy models)
```

Reading the map: (a) feeds *features into* every stage and is invisible at serving; (b) *replaces the decision logic* of one stage (re-ranking) and is visible at serving; (c) *redesigns the system that produces* every stage and never touches the request.

---

## 2. Reasoning Re-Rankers in Depth

### 2.1 From a fixed set objective to a learned reasoning policy

Re-ranking selects and orders the final $k$ items from $n$ candidates ([Note 06 §1](06_Reranking_and_Diversity.md)). Classical re-rankers optimize a **hand-crafted set objective** steered by one scalar $\theta$:

$$
\text{select } i = \arg\max_i  \underbrace{\theta \cdot \text{reward}_i}_{\text{value}} + \underbrace{(1-\theta)\cdot \text{diversity}_i}_{\text{MMR pairwise / DPP }\log\det}.
$$

A reasoning re-ranker discards the closed-form objective. The LLM reads the user's history and the candidate set — each item tagged with **Semantic IDs (语义 ID)** plus title/category metadata — and **reasons in natural language** about the right order, emitting a CoT trace and a structured (JSON) ranked list. The objective is no longer written down by a human; it is *learned* by RL. GR2's training pipeline:

1. **Semantic-ID mid-training (语义 ID 中训练).** Billions of raw item IDs would blow up the LLM vocabulary, so an RQ-VAE tokenizer (à la TIGER) maps each item to a short discrete code sequence with $\ge 99$% uniqueness, using a contrastive loss to encode co-engagement. The student LLM (e.g. Qwen3-8B) is mid-trained on a mixture of these SIDs interleaved with natural-language world knowledge, via next-token prediction — bridging recsys IDs into the LLM's linguistic/world-knowledge space.
2. **Reasoning SFT (推理监督微调).** A larger teacher LLM (e.g. Qwen3-32B) generates hierarchical CoT traces (history summary → category pattern → complementarity → candidate match) via re-ranking-specific prompts and **rejection sampling** (keep a trace only when its prediction matches the ground-truth next item, filtering hallucinated rationales). The student is SFT-ed on the curated traces, with decoupled loss weights $\lambda_r < \lambda_o$ so ranking accuracy is not drowned out by reasoning fluency.
3. **RL via DAPO (强化学习).** Decoupled Clip and Dynamic sAmpling Policy Optimization — a GRPO descendant that fixes entropy collapse and rollout-length bias — refines the policy against **verifiable rewards** designed for ranking (§2.2).

The payoff: GR2 surpasses the OneRec-Think baseline by **+2.4% Recall@5** and **+1.3% NDCG@5**, with ablations showing reasoning traces drive most of the gain and RL adds further lift on top of SFT.

### 2.2 RL reward design for ranking a slate

Ranking a slate is a good RL target because correctness is **verifiable**: we know the ground-truth next item, so we can score how much re-ranking promoted it. GR2 uses two terms:

- **Ranking reward** — how far the target item moved up:
$$
R_{\text{rank}} = \frac{r^{\mathcal{D}}_{s_{v_{n+1}}} - r^{o}_{s_{v_{n+1}}}}{|\mathcal{D}|},
$$
where $r^{\mathcal{D}}$ and $r^{o}$ are the target's ranks in the pre-ranked input and the re-ranked output. Positive when the target rose.
- **Format reward** $R_{\text{fmt}}$ — a binary check that the reasoning trace and the ranked list are parseable.

This RL framing is what distinguishes a *reasoning* re-ranker from a zero-shot prompted LLM: the model is optimized end-to-end for the ranking metric, not just asked to "please rank these."

### 2.3 Reward hacking (奖励作弊)

The cautionary tale of reasoning re-rankers. Naively summing $R = R_{\text{rank}} + \alpha R_{\text{fmt}}$ creates a degenerate optimum: the model can **just echo the input order verbatim** — producing valid format, collecting $R_{\text{fmt}}$, and never risking a ranking change. It thus learns to do nothing. GR2's fix is a **conditional format reward**: grant $R_{\text{fmt}}$ only when the re-ranking actually helped (or the target was already at rank 1):

$$
R = \begin{cases} R_{\text{rank}} + \alpha R_{\text{fmt}}, & \text{if } R_{\text{rank}} > 0 \text{ or } r^{\mathcal{D}}_{s_{v_{n+1}}} = 1, \\ R_{\text{rank}}, & \text{otherwise.} \end{cases}
$$

The general interview point: whenever you add an auxiliary reward (format, length, style) alongside the true objective, check there is no *trivial policy* that harvests the auxiliary reward while ignoring the objective. This is a recurring failure mode across all three agentic settings (it reappears as proxy-vs-north-star misalignment in §3.3).

### 2.4 Contrast with MMR / DPP (Note 06)

| | MMR / DPP (Note 06 §4–§6) | Reasoning re-ranker (GR2) |
|---|---|---|
| Objective | hand-crafted, closed-form ($\theta\cdot\text{reward} + (1-\theta)\cdot\text{diversity}$) | learned by RL; implicit |
| Diversity | explicit — pairwise $\max_j \text{sim}(i,j)$ (MMR) or $\log\det(A_S)$ (DPP) | *emergent* from reasoning over content semantics; no $\log\det$ term |
| Input signal | precomputed similarity/CLIP vectors + scalar rewards | item semantics (SIDs, titles, categories) + user history, in text |
| Compute | cheap linear algebra: DPP greedy is $O(n^2 d + nk^2)$ via incremental Cholesky | autoregressive decode of a CoT trace per request — orders of magnitude heavier |
| Tuning | human tunes $\theta$ against dwell-time proxies | RL tunes the policy against a verifiable ranking reward |
| Failure mode | $\theta$ mis-set; diversity term saturates for large $S$ | reward hacking; hallucinated reasoning |

The geometric intuition of Note 06 (diverse set ⇒ orthogonal unit vectors ⇒ large hyper-volume) does not vanish — it is just no longer computed explicitly. GR2 is viable mainly for **small slates** ($n$ in the tens) where world-knowledge reasoning beats vector geometry and the per-request LLM cost is tolerable; DPP remains the right tool when $n$ is large or latency is tight.

---

## 3. Agentic RecSys: Simulated Environments & Self-Evolving Pipelines

### 3.1 Why offline metrics and A/B are insufficient

Both classical evaluation modes have structural limits:

- **Offline metrics** (Recall@N, NDCG, counterfactual policy evaluation) are computed on *historical logs*, which carry **exposure bias (曝光偏差)**: you only observe feedback on items the old policy chose to show. A bold new policy that recommends genuinely different content has no log to score against, so offline metrics systematically *reinforce known patterns* and cannot credit the discovery of new interests.
- **Online A/B tests** measure the truth but have a **slow feedback loop** and expose **real users** to potentially bad experiences. The true objective — long-term user satisfaction / retention — is **latent, delayed, noisy, and sparse**, observed only on the order of $\Theta(\text{days})$–$\Theta(\text{weeks})$. You cannot iterate on a weeks-long signal at research speed.

This is precisely the evaluation gap that re-ranking inherits: in Note 06 we tune $\theta$ against offline dwell-time *proxies* because the real retention objective is unmeasurable per-request. Agentic methods attack the gap from two directions.

### 3.2 What a world model / simulated user gives you

**RecoWorld** provides a **simulated online environment** — a Gym-like sandbox for recommenders. Its dual-view architecture pits a **simulated user** (LLM) against an **agentic recommender** over **multi-turn** sessions, jointly optimizing retention:

- The simulated user reviews the recommended items, updates an internal "mindset," and — when sensing disengagement — issues a **reflective natural-language instruction** instead of silently churning ("show me something different but related," "what are people in the Bay Area watching?").
- The recommender adapts via reasoning traces, producing an updated list; the loop continues until the user exits. Engagement statistics (clicks, watch time, turns-per-session) become **pseudo-rewards** for multi-turn RL.

What the world model buys you:

1. **Cheaper-than-A/B iteration** on bold strategies without harming real users — test radically different policies in the sandbox first.
2. A reframing of re-ranking from a **one-shot static set-selection** (the entire premise of Note 06 — pick $k$ of $n$ once) into a **multi-turn interactive policy** where the user can *instruct* the system to re-diversify. The diversity/relevance trade-off that $\theta$ encodes statically becomes a *learned dynamic policy*: diversity emerges because monotony triggers a disengagement instruction, not because a $\log\det$ term penalizes it.
3. An **instruction-following** benchmark for recommenders — analogous to instruction-following for chat LLMs — plus use cases for creator strategy simulation and bandit-style new-user exploration.

The catch: simulated feedback **cannot perfectly replicate real interactions** (the sim-to-real gap). RecoWorld is a blueprint/position paper, not a deployed model; its validity must be checked against real datasets and human annotators, and LLM-simulating every user turn is computationally heavy.

### 3.3 Self-evolving pipelines: AutoML by LLM agents

Where RecoWorld supplies the *environment*, the **Self-Evolving Recommendation System** (deployed at YouTube) automates the *engineer*. It frames model improvement as **bi-level optimization**:

$$
\underbrace{\theta^{\ast}(\Phi) = \arg\min_\theta L_{\text{proxy}}(D; \theta, \Phi)}_{\text{lower level: train weights on a proxy reward}}, \qquad
\underbrace{\Phi^{\ast} = \arg\max_\Phi  \mathbb{E}\big[M(\theta^{\ast}(\Phi))\big]  \text{s.t. } G(\Phi) \le C}_{\text{upper level: choose meta-config for delayed north-star metric}}
$$

where $\Phi$ = {optimizer, architecture, reward definition} and $M$ = delayed business (north-star) metrics. Two synchronized loops run it:

- **Offline Agent (inner loop):** high-throughput hypothesis generation scored against fast **proxy metrics**.
- **Online Agent (outer loop):** validates the survivors against **delayed north-star metrics** in live production.

The agents (Gemini 2.5 Pro) act as **ML engineers**: they read production code, propose **structural mutations** (new activations like Swish/GELU, interaction layers like DCN/Transformer) and **novel reward functions** that encode nuanced concepts (intent, exploration). This goes strictly beyond classical AutoML/NAS, which can only *select from a fixed menu* of numerical hyperparameters and cannot *invent* new reward logic or topology. The result: several successful production launches at YouTube, beating hand-tuned baselines on both development velocity and model performance.

This is the most radical departure from Notes 06–07: instead of a human choosing MMR vs. DPP and hand-tuning $\theta$, **an LLM agent autonomously designs and deploys the objective** — including the diversity/exploration terms we built by hand. It operationalizes the Note 06 §7 observation that *diversity must be tuned for business metrics*: the outer loop optimizes directly for delayed retention, automating the manual $\theta$-tuning loop. The cost shifts from per-request latency to a **slow, expensive meta-optimization loop** bounded by how fast real-world retention signals arrive — and it demands strict safety guardrails, since the agents touch production code and **proxy-vs-north-star misalignment** is the §2.3 reward-hacking risk at the system level: there is no clean oracle for "user satisfaction."

---

## 4. Multimodal & Reasoning-Aligned Representations

Role (a) has a frontier of its own: not just *bigger* content encoders, but encoders **aligned to the recsys objective**. The naive recipe — cache frozen multimodal-LLM embeddings as extra item features — gives disappointingly small gains, for two reasons:

- **Representation unmatch (表征不匹配):** the LLM's pre-training objective (captioning, QA, image-text matching) is *not* the recsys objective (click prediction, "discover *new* items expanded from history" rather than "fetch look-alikes").
- **Representation unlearning (表征无法端到端学习):** frozen embeddings cannot update end-to-end with the downstream ranking task.

**QARM V2** ([Note 05 — Sequence Modeling](05_Sequence_Modeling.md), Expanded Reading) is the reasoning-aligned answer, and it shows roles (a) and (b) blending. It keeps the SIM/TWIN **GSU → ESU** long-sequence paradigm but upgrades the *similarity signal*:

- **GSU side — reasoning item alignment.** Fine-tune an LLM on contrastive item pairs mined from Swing (I2I) and two-tower (U2I) retrieval, *filtered by a reasoning LLM* (Qwen3) that drops hot-popular-biased and unrelated pairs (rejecting ~10% of I2I, ~70% of U2I). This yields **business-aligned LLM embeddings** for semantic retrieval of relevant historical sub-sequences — fixing representation unmatch.
- **ESU side — Res-KmeansFSQ.** Quantize those embeddings into multi-level **Semantic IDs** (residual K-means for coarse category/usage layers, Finite Scalar Quantization for the fine layer to cut codebook collisions). The SIDs are **learnable discrete features trained end-to-end** with the ranking model — fixing representation unlearning.

The takeaway for the spectrum: a content/user FM is only as good as its *alignment* to the downstream task. QARM V2's "reasoning-aligned multimodal embeddings feeding sequence modeling" is the encoder role done right, and its Semantic-ID machinery is exactly the substrate GR2's reasoner consumes in §2 — the two roles share the SID tokenization layer.

---

## 5. Trade-offs: When Is an LLM Worth It?

The single most important interview judgment is **knowing when *not* to use an LLM.** Frame it by where the cost lands:

- **Serving latency & cost.** A DPP greedy re-rank is $O(n^2 d + nk^2)$ of cheap linear algebra; autoregressive decoding of a multi-step reasoning trace from an 8B-parameter LLM *per request* is orders of magnitude heavier and adds real tail latency. Role (a) sidesteps this entirely by precomputing embeddings offline — which is why it is the most-deployed role. Role (b) is only viable for small slates and high-value surfaces. (For the serving/inference-efficiency techniques — distillation, quantization, KV-cache, batching — that make role (b) tractable, see [Note 12 — Serving & Inference Efficiency](12_Serving_and_Inference_Efficiency.md).)
- **Hallucination & reward hacking.** Reasoning traces can be ungrounded (mitigated by forcing SID citation + rejection sampling); RL policies exploit any loophole in the reward (§2.3); agentic systems chase proxy metrics that diverge from the north-star (§3.3). All three demand verifiable rewards and guardrails.
- **Evaluation difficulty.** The true objective (long-term satisfaction) is latent and delayed; offline metrics carry exposure bias and A/B is slow (§3.1). This is *why* simulated environments and bi-level optimization exist — but the sim-to-real gap means they are aids, not oracles.
- **When a cheap DLRM wins.** If the candidate set is large, latency is tight, the signal is well-captured by ID/CLIP embeddings, and there is no need for world-knowledge reasoning, a pointwise DLRM + DPP re-rank is faster, cheaper, more predictable, and easier to debug. Reach for an LLM when you specifically need (i) world knowledge / semantic generalization (cold-start, long-tail — Note 07), (ii) reasoning over content the DLRM can't see, (iii) natural-language instruction-following, or (iv) automated discovery of objectives/architectures beyond a human's iteration bandwidth.

A useful heuristic: **push the LLM as far off the request path as the task allows.** Encoder (offline) > reasoner (per-request) > agent (dev loop) is roughly the order of increasing experimental risk and the order in which the LLM's cost intrudes on serving — choose the leftmost role that solves your problem.

---

## 6. Key Insights

| Topic | Key point |
|---|---|
| Three roles | (a) **encoder/FM** — offline feature; (b) **reasoner/reranker** — on request path; (c) **agent** — in dev loop. Cost lands offline / per-request / in training respectively. |
| Role (a) examples | User FM (Coinbase, Note 05); content FM (Netflix MediaFM, Note 07); reasoning-aligned multimodal SIDs (QARM V2, Note 05). |
| Role (b) = GR2 | **SID mid-training → reasoning SFT → RL (DAPO)**. Beats OneRec-Think +2.4% Recall@5, +1.3% NDCG@5. |
| Semantic IDs | RQ-VAE tokenizer maps billions of raw IDs → short discrete codes ($\ge 99$% unique); avoids LLM vocab blowup; the shared substrate for GR2 and QARM V2. |
| Ranking reward | verifiable: how far re-ranking promoted the ground-truth target, $R_{\text{rank}} = (r^{\mathcal{D}} - r^{o})/|\mathcal{D}|$. |
| Reward hacking | naive $R_{\text{rank}}+R_{\text{fmt}}$ ⇒ model echoes input order to harvest format reward; fix = **conditional** format reward (grant only if ranking improved). |
| vs MMR/DPP | classical = hand-crafted closed-form objective + explicit $\log\det$ diversity, $O(n^2d+nk^2)$; reasoning re-ranker = learned RL policy, diversity emergent, far heavier; LLM viable only for small $n$. |
| Why not offline/A-B | offline = exposure bias (reinforces known patterns); A/B = slow + risks real users; true objective latent/delayed/sparse. |
| RecoWorld | dual-view simulated user ↔ agentic recommender, multi-turn RL on retention pseudo-rewards; user issues NL instructions; turns one-shot set-selection into a dynamic policy. Blueprint, sim-to-real gap. |
| Self-Evolving | bi-level optimization; **Offline Agent (inner, proxy metrics)** + **Online Agent (outer, delayed north-star)**; LLM agents act as MLEs authoring architectures/rewards; beyond AutoML/NAS; YouTube launches. |
| QARM V2 fixes | representation **unmatch** (LLM objective ≠ recsys) + **unlearning** (frozen, no end-to-end); fix via reasoning item alignment (GSU) + learnable Res-KmeansFSQ SIDs (ESU). |
| When to use an LLM | world knowledge, reasoning over unseen content, NL instructions, or auto-discovery. Otherwise prefer cheap DLRM + DPP. Push the LLM as far off the request path as possible. |

---

## 7. References

- Liang, M., Li, Y., Xu, J., Asadi, K., Liu, X., Gu, S., Rangadurai, K., Shyu, F., Wang, S., Yang, S., et al. (2026). *Generative reasoning re-ranker.* arXiv. https://arxiv.org/abs/2602.07774
- Liu, F., Lin, X., Yu, H., Wu, M., Wang, J., Zhang, Q., Zhao, Z., Xia, Y., Zhang, Y., Li, W., et al. (2025). *RecoWorld: Building simulated environments for agentic recommender systems.* arXiv. https://arxiv.org/abs/2509.10397
- Wang, H., Wu, Y., Chang, D., Wei, L., & Heldt, L. (2026). *Self-evolving recommendation system: End-to-end autonomous model optimization with LLM agents.* arXiv. https://arxiv.org/abs/2602.10226
- Xia, T., Zhang, J., Liu, Y., Dou, H., Yin, T., Cao, J., et al. (2026). *QARM V2: Quantitative alignment multi-modal recommendation for reasoning user sequence modeling.* arXiv. https://arxiv.org/abs/2602.08559
- Edezhath, R. (2026). *Scaling personalization with user foundation models.* Coinbase Engineering Blog.
- Saluja, A., Castro, S., Yan, B., & Rastogi, A. (2026). *MediaFM: The multimodal AI foundation for media understanding at Netflix.* Netflix Technology Blog.

## Pinterest in Practice (2024–2026)

Pinterest deploys LLMs **pragmatically** — pushed as far off the request path as possible (the §5 heuristic): as offline teachers/judges and for cheap multi-objective alignment, *not* as an online per-request reasoner.

### LLM-as-judge for multi-objective alignment (UPO)

> Badrinath, A., Agarwal, P., & Xu, J. (2025). *Unified preference optimization: Language model alignment beyond the preference frontier*. arXiv. https://arxiv.org/abs/2405.17956

UPO is Pinterest's recipe for aligning an LLM to **both** user preference **and** non-preferential "designer" objectives (toxicity, obscenity, reading level, 可读性) that are sparse or absent in the labeled preference data. It fuses a DPO/KTO-style **preference MLE** loss with an *offline-RL* **advantage-weighted MLE** auxiliary term — adding "~10 lines of code" and at most 0.4% of training time on top of KTO, ~3× cheaper than on-policy PPO. The key trick is that because the auxiliary term has no paired $y_w/y_l$, it sidesteps the MODPO sigmoid-gradient degeneracy that stalls binary multi-objective methods when a safety signal contradicts the preference label. Conceptually this is the LLM analogue of the **multi-objective ranking blend** (repins + clicks − hides, tuned on a Pareto front) that Pinterest already does in its DLRMs — letting designers "bolt on" safety/style constraints cheaply without an H100 mega-cluster. It is the cost-and-pragmatism end of the LLM-alignment story: a stable, offline, MLE-flavored method rather than expensive online RL.

### LLM-as-teacher for search relevance distillation

> Wang, H., Narayanan S, M., Gungor, O., Xu, Y., Kamath, K., Chalasani, R., Hazra, K. S., & Rao, J. (2024). *Improving Pinterest search relevance using large language models*. arXiv. https://arxiv.org/abs/2410.17152 (CIKM '24, 3rd Int'l Workshop on Industrial Recommendation Systems)

This is the canonical **"LLM as offline teacher, lightweight model online"** industrial pattern — exactly the §1.1 LLM-as-encoder/teacher role, where the expensive forward pass runs offline and serving stays cheap. A fine-tuned **Llama-3-8B cross-encoder** (qLoRA, 4-bit) scores (query, Pin) relevance on a 5-level scale and is **distilled into a lightweight feed-forward student** that serves online over precomputed SearchSAGE/PinSAGE/visual embeddings — resolving the accuracy-vs-latency split since the cross-encoder is far too slow to serve at Pinterest scale in real time. Two ideas make it work: (1) the visual problem is turned into a **text problem** via proxies — BLIP image captions, link/board titles, and high-engagement query tokens stand in for the image; (2) the teacher labels **billions** of unlabeled impressions, scaling training data ~100× and, because both teacher and logs are multilingual, giving **free cross-lingual generalization** to 45+ languages with zero new human labels (Non-US fulfillment +2.0% vs. US +0.7%). The trade-off is explicit: the served student (0.548) sits well below the teacher (0.602) — distillation narrows but does not close the gap, the price of staying within the online latency budget.
