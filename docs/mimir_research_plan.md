# Peptide Binder Generation — Research Plan

PDF version: [Mímir Research Plan](mimir_research_plan.pdf).

---

## Conceptual Foundation

### Why Binding is a Thermodynamic Problem

A peptide binds a receptor because the bound state minimizes free energy (ΔG = ΔH - TΔS). Binding is spontaneous when ΔG is negative — the enthalpic gain from atomic contacts (hydrogen bonds, van der Waals, salt bridges, hydrophobic burial) outweighs the entropic cost of losing conformational freedom.

This means binding is fundamentally physical. A model that generates binders must have internalized this physics, explicitly or implicitly.

### The Sequence → Structure → Energy Pipeline

The field is built on a cascade of proxies, each trading physical rigor for computational speed:

- **Sequencing is cheap, structure determination is expensive** → train AF2/Boltz-2 to predict structure from sequence
- **Geometry is cheap to predict, thermodynamics is expensive to compute** → use geometric confidence (iPSAE, pTM) as proxy for energy
- **Contact counting is cheap, MD simulation is expensive** → use PRODIGY/DeltaForge as proxy for ΔG

Structure predictors like Boltz-2 do not compute energy explicitly. They learned sequence→structure correlations from PDB data, which is an evolutionarily filtered sample of sequence space — biased toward stable, low free energy conformations. Energy is implicit in the training data, not computed by the model.

### What Matters for Binder Design

For binder design the relevant prediction is not single-chain folding but **co-folded complex prediction**. The peptide may have no stable fold in isolation — it may only fold upon binding. The relevant input to Boltz-2 is therefore two sequences simultaneously, not individual chains.

Geometric confidence metrics (iPSAE, pTM) and thermodynamic scores (ΔG from DeltaForge) are **orthogonal**. They measure different things:

- iPSAE: does the structure predictor find a coherent binding pose
- ΔG: is that pose energetically favorable

Filtering on either alone discards the best candidates of the other. Both are needed.

### Why ESM3 is the Right Foundation

ESM3 was pretrained on sequence, structure tokens (1D VQVAE-compressed), SASA, and secondary structure simultaneously across millions of proteins. Its embeddings implicitly encode thermodynamic plausibility — sequences that survived evolutionary selection are sequences that fold stably and function reliably. ESM3 already knows protein physics without ever explicitly computing energy.

This makes ESM3 a natural foundation: its implicit thermodynamic knowledge can serve as a training signal, and its multi-track architecture natively supports conditioning on any subset of tracks at inference.

---

## The Vinland Angle

### The Bias Problem in the Field

Generative peptide design methods are doubly biased:

1. Models are trained on PDB data, which overrepresents well-studied receptors (kinases, GPCRs, proteases, immune checkpoints)
2. Researchers input those same well-known receptors because those are the ones with known structures

The result: impressive throughput against a narrow slice of protein space.

### Vinland as a Solution

Vinland is a curated atlas of virus-human protein-protein interactions with annotated interacting domains. Viral peptides are experimentally validated binders — nature tested them under extreme evolutionary pressure. Viruses hijack essential but overlooked host proteins, mapping underexplored but biologically meaningful binding sites that pharma has not focused on.

Key properties:

- Peptides are experimentally validated (published literature, manually curated)
- Target proteins are known
- Binding sites on targets are **unknown** → to be determined via Boltz-2
- Receptor diversity is broad — viral coevolution explored protein space far beyond standard drug targets

### The Boltz-2 Pipeline for Vinland

Boltz-2 takes two sequences and outputs the co-folded complex with full 3D coordinates. For each Vinland viral peptide / human target pair:

```
Viral peptide sequence + Human target sequence → Boltz-2 → Co-folded complex → Binding site residues
```

The binding site is not an input — it is an **output**. This computationally maps where on each human target the viral peptide docks, revealing underexplored pockets for therapeutic targeting.

The downstream pipeline:

```
Vinland pairs → Boltz-2 → binding sites on underexplored targets → train Mimir → generate novel binders for those sites
```

---

## Core Research Hypotheses

**Hypothesis 1 — Vinland + Boltz-2 unlocks new receptor space.**
Viral coevolution mapped underexplored but validated binding sites on human proteins. Boltz-2 can computationally reveal those sites well enough to train on. The diversity of viral targets breaks the narrow receptor bias of existing methods and training datasets.

**Hypothesis 2 — ESM3 already knows enough thermodynamics that fine-tuning on complexes is sufficient.**
ESM3's pretraining on evolutionary data gives it implicit physical knowledge of what sequences are compatible with what structures. Fine-tuning is not teaching physics from scratch — it is redirecting existing knowledge toward binding context between two chains.

_Boltz-2 reliability is a practical dependency, not a hypothesis. It is widely used in serious research and will be validated empirically._

---

## Dataset

### Overview

Three tiers with distinct roles:

| Tier | Source                                               | Role                                           | Quality   |
| ---- | ---------------------------------------------------- | ---------------------------------------------- | --------- |
| 1    | PDB (all species)                                    | Ground truth geometry, experimental validation | Highest   |
| 2    | AlphaFold Database complexes (new release)           | Scale and species diversity                    | Predicted |
| 3    | Vinland viral peptides + Boltz-2 predicted complexes | Receptor diversity, underexplored pockets      | Predicted |

Each tier contributes something the others cannot. PDB is ground truth but narrow. AlphaFold adds scale. Vinland adds diversity across protein space that neither PDB nor AlphaFold alone provides.

### AlphaFold Complex Release (March 2026)

_Reference: https://www.embl.org/news/science-technology/first-complexes-alphafold-database/_

The March 2026 AlphaFold Database update is a major resource for this project:

- 30 million complexes calculated in total
- 1.7 million high-confidence homodimer predictions immediately available
- 18 million lower-confidence homodimers available for bulk download
- Heterodimers currently being analysed — high-confidence predictions to be added in coming months
- Focuses on 20 most studied species including humans + WHO priority pathogens
- Would have required ~17 million GPU hours to recreate independently

**Implications for this project:**

- Homodimers immediately useful — many relevant receptor architectures (KIT, VEGF-A etc.) are homodimers
- WHO priority pathogens directly overlap with Vinland's viral focus
- Heterodimer release (coming months) will be the most relevant tier for peptide-receptor pairs
- Complementary to Vinland data — requires dedicated curation efforts

### Dataset Curation — A Research Workstream in Itself

Dataset curation is not preprocessing — it is a significant research contribution. The choices made on quality thresholds, diversity sampling, and tier weighting directly determine model generalization. This requires its own versioned pipeline.

**Quality filtering:**

- PDB: use experimental resolution as quality proxy
- AlphaFold complexes: pLDDT per chain + ipTM threshold for interface confidence. 1.7M high-confidence homodimers already filtered; heterodimers will need custom thresholds
- Vinland: Boltz-2 iPSAE threshold on predicted complexes

**Diversity sampling:**

- 18M homodimers is enormous — naive inclusion drowns out Vinland signal
- Cluster by receptor family and/or sequence identity cutoff
- Balance representation across protein classes — prevent kinase/GPCR dominance

**Redundancy removal:**

- PDB and AlphaFold will overlap significantly on receptor structures
- Same receptor, different binders: keep (useful diversity)
- Same receptor, same binder from multiple sources: collapse to highest quality instance

**Practical considerations:**

- Oversample PDB examples (highest quality)
- Weight Vinland carefully — smaller volume but uniquely diverse signal
- Per complex encode: receptor (sequence + 1D structure tokens + 3D coordinates + SASA) and binder (sequence + 1D structure tokens)
- Prefer a small high-quality dataset that can be scaled over a large noisy one
- Focus on well-defined, high-confidence samples

---

## Training Strategy

### Core Design Principles

1. **Sequence is primary, structure is secondary.** One peptide sequence can adopt multiple structures depending on receptor context. The model generates sequences; structure is inferred downstream.
2. **ESM3's implicit thermodynamic knowledge is the training signal.** No explicit energy computation. The binder structure track provides geometric context that ESM3 uses to evaluate sequence guesses against known physical constraints.
3. **One training session, three-tier dataset.** No curriculum, no separate Vinland fine-tuning phase. The population diversity handles distribution learning.

### Input/Output During Training

| Track                        | During Training              | Penalized?          |
| ---------------------------- | ---------------------------- | ------------------- |
| Receptor sequence            | Fully visible                | No                  |
| Receptor 1D structure tokens | Fully visible                | No                  |
| Receptor 3D coordinates      | Fully visible                | No                  |
| Receptor SASA                | Fully visible                | No                  |
| Binder 1D structure tokens   | Visible with dropout masking | No                  |
| Binder sequence              | MLM 0–75% masking            | **Yes — only loss** |

The binder structure track is **additional context** — like SASA — giving the model as much information as possible to make good sequence guesses. It is not a prediction target. Dropout masking ensures the model does not become dependent on it, since it will be absent at inference.

### Masking Strategy

- **Binder sequence masking rate:** uniform 0–75%
- **0% lower bound** (vs Mimir v2's 25%) handles the first inference iteration starting from fully masked input
- **Forced full-mask anchors:** per epoch, per receptor, force at least one training example with binder sequence fully masked. Implemented as a dataloader constraint — guarantees coverage regardless of sampling.
- **Binder structure dropout:** random partial masking across samples, variable rate. Never penalized, purely for conditioning robustness.

### Why This Works as Thermodynamic Supervision

The binder's 1D structure track comes from experimentally validated or Boltz-2 predicted complexes — conformations that work. ESM3 learned which amino acids are physically plausible in which structural contexts. A bad sequence guess incompatible with the visible conformation produces anomalous ESM3 representations → higher loss. Thermodynamic signal emerges from ESM3's evolutionary knowledge meeting the validated conformation, with no explicit energy computation.

---

## Inference

### Pass 1 — Generate Binder Sequence

```
Input:  Receptor (all tracks, fully visible)
        Binder sequence (fully masked)
        Binder structure (fully masked)

Output: Binder sequence
```

The model generates a peptide sequence conditioned on receptor geometry. Iterative unmasking from fully masked state, same as standard discrete diffusion inference.

### Pass 2 — Infer Binder Structure _(cherry on the cake)_

```
Input:  Receptor (all tracks, fully visible)
        Binder sequence (generated in Pass 1, fully visible)
        Binder structure (fully masked)

Output: Binder 1D structure tokens
```

Optional second pass leveraging ESM3's native sequence→structure inference in complex context. The alternative is simply running Boltz-2 on the generated sequence, which is always available as a fallback.

---

## Publication Checkpoints

Each checkpoint is independently publishable. If a later stage fails, earlier stages still produce scientific contributions.

**Checkpoint 1 — Vinland Binding Site Atlas**
Run Boltz-2 on all Vinland viral peptide / human target pairs. Analyze predicted binding sites, show they are structurally distinct from canonical drug targets. Pure bioinformatics, no model training required. Standalone resource paper. Could attract wet lab collaborators for experimental validation of specific sites.

**Checkpoint 2 — ESM3 Baseline on Complexes**
Before or alongside fine-tuning, benchmark ESM3's native ability to handle peptide-receptor complexes. Quantifies what fine-tuning actually contributes and establishes a methodological baseline interesting to the field regardless of downstream results.

**Checkpoint 3 — Experimental Validation**
One SPR or BLI confirmed hit against a novel Vinland-derived receptor. Proves the full pipeline end to end — from viral coevolution signal through computational binding site discovery to generative model to experimental binder. This is the final checkpoint that validates everything.

---

## Validation Strategy

- Score generated complexes with Boltz-2 + DeltaForge (iPSAE + ΔG dual-metric)
- Compare predicted binding sites from Vinland Boltz-2 runs against any available mutagenesis or functional data for those target proteins
- Experimental validation: SPR/BLI on prioritized candidates against Vinland targets

---

## Open Questions

1. How well does ESM3's structure inference generalize to peptide-receptor complexes it was not pretrained on?
2. Does binder structure dropout rate need tuning per tier (PDB vs predicted structures)?
3. What is the minimum Vinland dataset size needed to shift receptor diversity meaningfully?
4. Can the pairwise thermodynamic ranking insight (fixed receptor + precomputed substitution impact) be used as a post-training reranker?
5. What confidence thresholds to apply to AlphaFold heterodimer release for training inclusion?
6. How to balance AlphaFold homodimers scale against Vinland pairs without losing the viral diversity signal?
