# Mimir v2 — Design Document

## AI-Driven Peptide Binder Generation for Drug Discovery

PDF version: [Mímir v2 Design Document](mimir_v2_design.pdf).

---

## Table of Contents

1. [Origins — The Vinland Database](#1-origins--the-vinland-database)
2. [Project Overview](#2-project-overview)
3. [From v1 to v2 — What Changed and Why](#3-from-v1-to-v2--what-changed-and-why)
4. [Foundation Model — ESM3](#4-foundation-model--esm3)
5. [Data](#5-data)
6. [The Protein Fingerprint — Mimir v2 Fingerprinting Protocol](#6-the-protein-fingerprint--mimir-v2-fingerprinting-protocol)
7. [Token Layout](#7-token-layout)
8. [Binder Encoding and Masking Strategy](#8-binder-encoding-and-masking-strategy)
9. [Generalization Strategy — Domain-Level Learning](#9-generalization-strategy--domain-level-learning)
10. [Emergent Capability — Structure Inference for Sequence-Only Binders](#10-emergent-capability--structure-inference-for-sequence-only-binders)
11. [Evaluation Strategy](#11-evaluation-strategy)
12. [Hardware and Training Efficiency](#12-hardware-and-training-efficiency)
13. [Summary of Key Design Decisions](#13-summary-of-key-design-decisions)

---

## 1. Origins — The Vinland Database

**Vinland** (vinland.network) is one of the largest manually curated databases of protein-protein interactions, covering both human-human and human-virus interactions. Building it required years of systematic literature mining and expert biological annotation — recording which proteins interact, and when available, the binding sequence that mediates the interaction. These sequences are real, experimentally supported peptide binders extracted from biology. The question they raise: can a model learn from them to propose binders that nature never explored?

The human-virus dimension of Vinland is particularly valuable for drug discovery. Viruses are, in a sense, the product of nature's own drug discovery process. Over millions of years of coevolution with their hosts, they have evolved short peptide sequences that bind human proteins with remarkable specificity — hijacking cellular machinery, evading immune responses, manipulating signaling pathways. Each of these viral binding sequences represents an optimized solution to the problem of binding a human target. They are nature's answer to a question that medicinal chemists ask every day.

Vinland captures this evolutionary knowledge at scale. The sequences it contains have been validated by evolution itself — they work, or the viruses carrying them would not have survived.

But evolution is not exhaustive. It explores only the sequences that arise by mutation and are selected for fitness. The vast majority of sequence space — including potentially superior binders for therapeutically important targets — has never been touched. This is where Mimir intervenes. Trained on Vinland's curated interactions, Mimir learns the grammar of protein binding and uses it to propose new peptide sequences: plausible binders that evolution never produced, for targets that may have no known natural binder at all.

---

## 2. Project Overview

Mimir is a generative AI model that produces peptide binding sequences for human protein targets. It is fine-tuned from ESM3 using the Vinland dataset augmented with PDB-derived structural data.

Mimir v2 is a major architectural leap over v1. Where v1 represented each target protein as a single learned embedding token, v2 encodes the full 3D structural surface of the target, enabling true generalization to proteins never seen during training.

---

## 3. From v1 to v2 — What Changed and Why

### Mimir v1

- 15,000 training associations between protein targets and known binders
- ~5,500 unique target proteins, each represented by a **single learned token** (its UniProt accession ID)
- Binders up to 512 amino acids
- The model essentially learned a lookup table: token → binder sequence distribution
- Built on ESM3 1.4B fine-tuned with LoRA
- Converged around 100 training epochs

**Proof of concept result:** In a blind test, Mimir v1 generated peptides for two protein families (one with an SH3 domain, one with a PDZ domain). Without any label information, the generated sequences were distinguishable enough that the two lists could be correctly assigned to their respective domain families. This demonstrated that even with minimal target representation, the model learned biologically meaningful binding signatures.

### Mimir v2

- 30,000 training associations
- ~7,700 unique target proteins, each represented by a **rich structural fingerprint** (sequence + 3D structure + solvent accessibility)
- Binders up to 96 amino acids
- The model must learn to interpret a structural surface description and generate compatible binders
- Built on ESM3 1.4B fine-tuned with LoRA (r=16, alpha=32, dropout=0.1)
- Expected convergence: 150–300 epochs

The key insight driving v2: a single token per protein has no transferable information. Two structurally similar proteins have unrelated tokens. With a structural fingerprint, two proteins sharing a domain family will have similar fingerprints in ESM3's representation space, enabling **generalization by structural similarity**.

---

## 4. Foundation Model — ESM3

Mimir v2 is a LoRA fine-tune of **ESM3 1.4B** (EvolutionaryScale). ESM3 is a multimodal protein language model that natively handles five biological tracks simultaneously, of which Mimir v2 uses three:

- **Sequence track** — amino acid identity at each position
- **Structure track** — 3D structural tokens derived from coordinates
- **SASA track** — Solvent Accessible Surface Area per residue

The remaining two ESM3 tracks (secondary structure and function annotations) are masked and not used in v2.

ESM3 has two complementary mechanisms for spatial reasoning. The standard attention layers use **RoPE (Rotary Position Embeddings)** for sequential position encoding. More importantly for Mimir v2, the first transformer layer uses **geometric attention** — instead of learned positional encodings, it operates directly on 3D backbone coordinates (N, Cα, C atoms) to compute rotation-invariant affine frames between residues. This means the model perceives the true spatial layout of a protein surface through its actual geometry, not through a proxy positional index.

ESM3 also natively supports **multi-chain inputs** via a `chain_id` tensor. Residues with different chain IDs are separated in the geometric attention layer, allowing the model to reason about two independent molecular entities simultaneously — exactly the setting Mimir v2 requires for target-conditioned binder generation.

LoRA fine-tuning modifies only a small set of adapter weights on top of the frozen ESM3 backbone, keeping checkpoints small and training efficient.

---

## 5. Data

### 5.1 Training Associations

30,000 pairs of (target protein UniProt accession, binder sequence/structure). Sources:

- **PDB-derived binders (~5,500):** experimentally resolved structures of peptide-protein complexes. These provide sequence + 3D structure for the binder.
- **Vinland binders (~25,000):** binding sequences curated from the literature without experimental 3D structure. These provide sequence only for the binder.

Only binders of 96 amino acids or fewer are included. The 82% ratio of sequence-only binders is a data reality, not a design choice — the alternative would be to discard 25,000 validated binding sequences and train on 5,500 only, which would be a worse decision. The asymmetric supervision strategy (section 8) extracts what each source can contribute.

As with any literature-derived database, well-studied protein families — kinases, p53, nuclear receptors — are overrepresented in Vinland. The model will generalize most readily to these families. This is a known property of the dataset, not a correctable flaw.

### 5.2 Target Proteins

~7,700 human proteins from the training set. All target structural information is sourced from the **AlphaFold Database v6**, human proteome. AlphaFold v6 provides the best available structural predictions at proteome scale — using only experimentally resolved structures would reduce coverage to a small fraction of available targets.

Each AlphaFold structure provides:

- Amino acid sequence
- Predicted 3D coordinates
- Per-residue pLDDT confidence score (0–100)

SASA is computed from the AlphaFold structure and normalized as **relative SASA (rSASA)** — the fraction of surface exposed relative to a fully exposed reference value for each amino acid type.

### 5.3 Scope

Mimir v2 targets proteins with a structured, accessible surface. Intrinsically disordered proteins (IDPs) are out of scope by design — the fingerprinting protocol requires structurally confident residues and defined surface geometry. This covers the vast majority of exploitable drug discovery targets. IDPs are a different problem for a future version.

---

## 6. The Protein Fingerprint — Mimir v2 Fingerprinting Protocol

Many human proteins have hundreds or thousands of amino acids and cannot be fed directly into the model. The fingerprint protocol compresses each protein to its most biologically relevant surface residues through four sequential steps, producing a representation capped at **280 tokens**.

**Step 1 — Rigidity Gate:** Remove all residues with pLDDT < 70. AlphaFold's per-residue confidence score is low for disordered loops, flexible tails, and regions it cannot predict reliably. These regions are structurally undefined and irrelevant for binding — a peptide binder needs a stable surface to dock against.

**Step 2 — Surface Gate (adaptive):** Compute a **smoothed rSASA** for each surviving residue using a sliding window of 15 residues (the residue itself plus 7 on each side). Smoothing reasons in terms of regions rather than individual residues — a single slightly-buried residue surrounded by exposed neighbours is retained, while a genuine buried stretch is removed as a coherent block. The rSASA threshold starts at 0.01 and increases by 0.01 iteratively until the surviving residues fit within the 280-token window. Buried residues in the hydrophobic core are physically inaccessible to a peptide binder and carry no useful signal.

**Step 3 — Survival Check:** If fewer than 15 residues survive both gates, discard this protein entirely from training. It lacks sufficient surface signal to be meaningful.

**Step 4 — Restore Sequence Order:** Re-sort the surviving residues by their original sequence position IDs. This preserves the correct sequential order for the sequence track. The 3D coordinates of each surviving residue carry the true spatial information — gaps where buried residues were removed are implicit in the coordinate geometry, not in positional indices.

**Result:** A fingerprint of at most 280 tokens representing the structured, accessible surface of the protein, with original position IDs intact. Concave pockets and grooves — where rSASA is lower but above the adaptive threshold — are retained alongside flat exposed patches. High-affinity binders preferentially target pockets, so their retention is biologically important.

The 280-token cap is grounded in both biology and data. The longest single-fragment AlphaFold v6 structures reach approximately 2,800 residues — so 280 tokens guarantees at worst 1-in-10 residue resolution on the largest proteins in the dataset. In practice the resolution is much finer, since the two filters remove the majority of residues before the cap is reached. Proteins exceeding ~2,800 residues are stored as multiple ~1,400-residue fragments in AlphaFold v6 and are excluded from v2 — the cap handles all included proteins without truncation under normal filtering conditions.

---

## 7. Token Layout

Every training example is presented to ESM3 as two chains — the target fingerprint and the binder — separated by a native chainbreak token. No custom tokens are added to the vocabulary.

```
[BOS] + [protein fingerprint] + [CHAINBREAK] + [binder] + [EOS]
```

| Component           | Max tokens | Tracks                               |
| ------------------- | ---------- | ------------------------------------ |
| BOS                 | 1          | sequence + structure                 |
| Protein fingerprint | 280        | sequence + structure + SASA + coords |
| CHAINBREAK          | 1          | sequence + structure                 |
| Binder              | 96         | see section 8                        |
| EOS                 | 1          | sequence + structure                 |
| **Total max**       | **379**    |                                      |

Bucket-based batching groups examples by similar total length and pads each batch to the nearest multiple of 64 above the longest sequence in that batch, minimizing wasted compute while respecting H100 tensor core alignment.

### 7.1 The Chainbreak Token

ESM3 natively defines `SEQUENCE_CHAINBREAK (31)` and `STRUCTURE_CHAINBREAK (4100)` for separating multiple chains in a single input. These are placed at position N+1, between the fingerprint and the binder. No new tokens are added to the vocabulary — v1 required ~5,000 UniProt accession tokens, v2 requires zero.

### 7.2 Two Chains, Two Spaces

The fingerprint and the binder are assigned different `chain_id` values (1 and 2 respectively). In ESM3's geometric attention layer, residues with different chain IDs cannot attend to each other geometrically — the model treats them as two independent molecules in separate spatial frames. In all 47 remaining standard attention layers, full cross-chain attention operates freely, allowing the binder to be conditioned on the fingerprint.

This is the correct physical picture: the target and the binder are two separate molecules. Their 3D coordinates exist in independent reference frames. The geometric attention respects this separation, while the standard attention layers learn the conditioning relationship between them.

### 7.3 Training vs Inference — The Fingerprint Distinction

The fingerprint protocol compresses the entire exposed surface of a target protein into a single chain. This is a **training compromise** — because we do not know which specific domain or surface patch each Vinland or PDB binder targets, we include all exposed regions to ensure the relevant surface is always present.

At inference, this constraint disappears. A biologist targeting a specific domain of PIK3R1 does not need to input the full protein fingerprint — they input only the SH3 domain, or only the iSH2 domain, or any surface patch of interest. The model has no expectation about fingerprint completeness. Any structured, exposed surface fragment is a valid input.

This also enables a direct transfer learning validation: generate binders for the SH3 domain fingerprint of PIK3R1, then for the iSH2 domain fingerprint of PIK3R1. If the two lists are statistically distinct, the model learned domain-level binding logic from the full-fingerprint training data — transfer confirmed.

---

## 8. Binder Encoding and Masking Strategy

### 8.1 Binders with Structure (PDB-derived, ~5,500)

Available tracks: sequence + structure tokens. 3D coordinates and SASA are **not provided** for the binder — at inference time the binder does not exist yet and neither is available. Providing them during training would create a training/inference mismatch.

Structure tokens are computed from the PDB crystal coordinates via the ESM3 structure encoder. Positions where coordinates are missing (nan) receive token 2246 (the encoder's native output for undefined structure) and are excluded from the loss.

During training, sequence and structure tracks are independently partially masked, each with its own mask rate sampled independently between 25–75%. Loss is computed on masked sequence + masked structure tokens (excluding positions with token 2246), boosted by `1.0 + boost_ratio * log(N + 1)` where N is the total number of tokens participating in the loss. Heavily masked samples receive a higher weight — at inference the binder starts fully masked, so the model must learn to generate from scratch.

### 8.2 Binders without Structure (Vinland, ~25,000)

Available tracks: sequence only. Structure track is set to `STRUCTURE_MASK (4096)` throughout and excluded from loss.

During training, the sequence track is partially masked (25–75%). Loss is computed on masked sequence tokens only, boosted by `1.0 + boost_ratio * log(N + 1)` where N is the number of masked sequence positions.

### 8.3 Why This Asymmetry Works

This is multi-task learning with asymmetric supervision. The 5,500 PDB binders teach the model the relationship between a target surface and binder 3D structure. The 25,000 Vinland binders teach the model the breadth of sequence space for binders across many targets. Both datasets contribute what they uniquely can. The shared protein fingerprint representation ties them together in a unified learned space.

### 8.4 Masking Rate and Boost

Sequence and structure tracks are masked independently, each with a rate sampled uniformly between 25–75%. Decoupled masking allows asymmetric difficulty — e.g. low sequence noise with high structure noise in the same sample — preparing the model for the full range of inference conditions, including the case where a known sequence is provided and only structure needs to be predicted.

Samples with more masked tokens receive a higher loss weight via a logarithmic boost. This pushes the model to perform well on heavily masked examples, which best approximates the full-mask inference condition where the binder must be generated entirely from scratch.

---

## 9. Generalization Strategy — Domain-Level Learning

### 9.1 The Core Argument

The human proteome is not 20,000 independent proteins. It is a relatively small set of structural domain families recombined in different arrangements. ~7,700 training proteins likely cover the domain space far more completely than the raw count suggests.

Proteins not in the training set are typically novel combinations of domains that _are_ represented in the training set. Mimir v2 generalizes at the **domain level**, not the protein level.

### 9.2 How the Model Learns Domain-Level Associations

During training, each example pairs a full protein fingerprint with a known binder. We do not know which specific domain that binder engages — the fingerprint compresses the entire exposed surface, and literature annotations rarely specify the exact binding site.

This is the key mechanism. Consider two training proteins: protein 1 has domains (X, A) with binder list 1, and protein 2 has domains (X, B) with binder list 2. Peptide motifs common to both lists are associated with X. Motifs only in list 1 are associated with A. Motifs only in list 2 are associated with B. The association works in both directions simultaneously: shared domains concentrate shared motifs, differing domains discriminate the rest. No domain labels are needed — the combinatorial diversity of the training set is the signal.

At inference, inputting a single domain fingerprint is sufficient. The model learned domain-level associations from the full fingerprints — the isolated domain is simply a cleaner, more targeted version of what it already knows.

### 9.3 Evidence from v1

The blind SH3/PDZ test demonstrated this generalisation principle even with minimal target representation — a single learned token per protein, with no structural information at all. V2's structural fingerprint makes the mechanism far more explicit and powerful: two proteins sharing a domain will have geometrically similar fingerprint regions, and the model's learned associations transfer directly.

The clearest validation of domain-level transfer is the PIK3R1 test: generate binders for the SH3 domain fingerprint, then for the iSH2 domain fingerprint. Same protein, different domains. If the two lists are statistically distinct, the model did not memorize protein-level associations — it learned domain-level ones.

---

## 10. Emergent Capability — Structure Inference for Sequence-Only Binders

A valuable capability that falls out of the training design: after training, Mimir v2 can **infer the bound-state structure** of any binder for which only sequence is known.

Procedure: provide the protein fingerprint + the full unmasked binder sequence on the sequence track, mask the structure track of the binder, run a forward pass. The model predicts the binder structure conditioned on the target surface and the known sequence.

This is qualitatively different from standalone structure predictors like ESMFold, which predict structure in isolation. Mimir v2 predicts **structure in the context of a specific binding partner** — the predicted conformation is biologically contextualized.

Practical implication: the 25,000 Vinland binders in the training set can be retroactively assigned predicted bound-state structures, enriching the dataset for future versions.

---

## 11. Evaluation Strategy

Standard train/test methodology does not apply cleanly to generative models for biological sequences. There is no ground-truth output to compare against — asking whether a generated peptide matches a known binder exactly is meaningless. The model is supposed to generate _novel_ sequences. Evaluation therefore has three distinct phases with different purposes, tools, and datasets.

### 11.1 Training Metrics — Health Monitoring Only

During training, standard MLM metrics are tracked per epoch:

- **Accuracy** — fraction of masked tokens correctly reconstructed, reported separately for sequence track, structure track (binders with structure), and sequence-only binders
- **Perplexity** — exp(mean cross-entropy), model confidence measure
- **Learning rate** — to confirm warmup and decay are applied correctly

These metrics serve one purpose: detecting training problems (divergence, mode collapse, plateau too early). They are not evaluation metrics. A model with high training accuracy on seen proteins is not necessarily a good binder generator for unseen ones.

Expected ranges based on v1 (55% accuracy, perplexity 5.5): v2 should reach 65–70% overall accuracy and perplexity ~3–4 at convergence.

### 11.2 Validation — Selecting the Best Checkpoint

**Purpose:** identify which training epoch produced the best generalizing model.

**Dataset:** a carefully selected subset of the ~12,000 human proteins not present in the training set. Selection is done by clustering the full human proteome by sequence and structural similarity, then sampling representative proteins from clusters not covered by training proteins. This ensures the validation set is both diverse and genuinely unseen — not just proteins that happen to be missing from Vinland but are structurally near-identical to training proteins.

**Protocol:**

1. For each candidate checkpoint, generate N binder sequences per validation protein
2. Score all generated protein/peptide pairs with a **fast scoring judge** (e.g., AutoDock Vina or equivalent rapid docking algorithm)
3. Compute relative z-scores across checkpoints for each validation protein
4. The checkpoint with the best aggregate z-score across the validation set is selected

**Key properties of validation:**

- Relative comparison only — not absolute affinity measurement
- The known binder for each validation protein (where available) is scored as a reference baseline
- Fast judge is sufficient because the goal is checkpoint ranking, not precise affinity estimation
- Validation proteins are never used for any other purpose — they are frozen as a comparison tool

### 11.3 Test — In Silico Confrontation with Reality

**Purpose:** assess whether the selected model generates binders that are plausibly active, using the highest-quality in silico tools available before wet lab commitment.

**Dataset:** a small set of protein/peptide pairs that are:

- Not in the training set
- Experimentally validated (the peptide is known to bind the protein with measured affinity)
- Provided by the biology team — ideally including recent interactions not yet in Vinland (curated data has a lag of 1–2 years)

The known experimental binder for each test protein serves as the gold standard baseline.

**Protocol:**

1. Generate top candidates from the selected model for each test protein
2. Score generated candidates and the known experimental binder using a **powerful scoring judge** — e.g., AlphaFold3 complex prediction for peptide-protein pairs, which provides predicted complex structure and confidence scores
3. Compare generated candidate scores to the experimental binder score on the same target
4. If generated candidates score comparably to or better than the known binder on targets the model has never seen, this is strong in silico evidence of generalization

**Key properties of test:**

- Performed once, after checkpoint selection from validation
- Powerful judge justified by small dataset size — expensive tools are tractable at this scale
- Experimental binder as baseline makes the comparison interpretable regardless of absolute score calibration
- Test set is never touched until the model is selected — preserving its integrity as an unseen benchmark

### 11.4 Experimental Validation — Gold Standard

Synthesize top in silico candidates for selected test proteins. Test binding in vitro using appropriate assays (SPR, ITC, or target-specific biochemical assay). A confirmed experimental hit is the ultimate proof of concept and answers all theoretical critiques simultaneously.

---

## 12. Hardware and Training Efficiency

Training runs on a single H100 80GB GPU. Key efficiency decisions:

- **BFloat16** throughout — halves VRAM vs float32, H100 native BF16 tensor cores
- **Flash Attention 2** — O(L) memory attention, significant speedup on H100
- **Gradient checkpointing** — recomputes activations during backward pass, 4–8x activation VRAM reduction
- **torch.compile** — 20–40% speedup via kernel fusion on H100
- **Bucket-based batching** — groups similar-length sequences, pads to nearest multiple of 64
- **8-bit AdamW** (bitsandbytes) — 4x less optimizer state memory, proven in v1
- **LoRA r=16** — doubled from v1's r=8 to handle richer input representation
- No new tokens vs ~5,000 UniProt tokens in v1 — zero embedding overhead

Expected training time: 2–3 minutes per epoch, convergence at 150–300 epochs, total wall time well within a single H100 rental window.

---

## 13. Summary of Key Design Decisions

| Decision                            | Rationale                                                                           |
| ----------------------------------- | ----------------------------------------------------------------------------------- |
| ESM3 as backbone                    | Native 5-track protein model, 3 tracks used                                         |
| LoRA r=16, alpha=32                 | Doubled rank from v1 for richer input; alpha carried from v1                        |
| AlphaFold v6 for targets            | Best available structures at proteome scale                                         |
| pLDDT ≥ 70 gate                     | Only trust structurally confident regions                                           |
| Smoothed rSASA adaptive gate        | Removes buried regions coherently, fits 280 tokens                                  |
| 3D backbone coords for fingerprint  | Geometric attention uses true spatial layout, not positional proxies                |
| Native chainbreak token             | No vocabulary extension needed — ESM3 built for multi-chain                         |
| chain_id 1 and 2                    | Separates molecules in geometric attention, full conditioning in standard attention |
| SASA and coords withheld for binder | Avoids training/inference mismatch                                                  |
| Asymmetric loss                     | Each data source supervised on what it can contribute                               |
| 25–75% masking rate                 | Prepares model for full-mask inference condition                                    |
| Validation by clustering            | Representative unseen proteins, not random holdout                                  |
| Fast judge for validation           | Checkpoint ranking only — relative comparison                                       |
| Powerful judge for test             | Small dataset justifies expensive tools                                             |
| Experimental binder as baseline     | Makes test scores interpretable regardless of judge calibration                     |
| 96 AA binder cap                    | Scope: short peptide scaffolds for drug discovery                                   |

---

_This document incorporates design decisions, responses to critique, and the distinction between validation (checkpoint selection) and test (in silico confrontation with experimentally validated pairs). Intended as reference for biological collaborators and AI assistants implementing the training infrastructure._
