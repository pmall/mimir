# Mímir v2 Post-Mortem: Peptide Binder Generation with ESM3

Full technical design document: [Mímir v2 Design Document](docs/mimir_v2_design.md).

Future research plan: [Mímir Research Plan](docs/mimir_research_plan.md).

## The Project

Mímir is a two-version attempt to build a generative peptide binder model by fine-tuning ESM3 on the [Vinland](https://vinland.network) protein-peptide association database.

**V1** used a single learned embedding token per protein target. The model learned to associate protein identities with their known binder distributions but had no structural information and no mechanism to discriminate between binding domains on the same protein.

**V2** replaced the identity token with a 3D structural **fingerprint**: surface-exposed, structurally rigid residues extracted from AlphaFold predictions, fed to ESM3 as a separate chain alongside the masked binder peptide. The core hypothesis was that ESM3's geometric understanding would enable **transfer learning** — associating shared surface geometries across different proteins with compatible peptide motifs, and generalizing to unseen targets.

---

## What Was Learned

### Understanding ESM3

**Multi-track input design.** ESM3 simultaneously processes sequence, structure tokens, SASA, and 3D coordinates. Understanding what each track carries, how to mask them independently, and what the model expects at inference vs. training required reading the source directly rather than relying on documentation.

**ESM3's 1D structure encoder.** ESM3 encodes 3D structure into a discrete token vocabulary via a VQ-VAE. Understanding how raw coordinates are compressed into structure tokens, what information is preserved, and the implications for supervising the structure track was essential for interpreting training behavior — including why the structure track improved far more slowly than the sequence track.

**Geometric attention.** ESM3 encodes inter-residue geometry via relative distances within a chain. Understanding this mechanism — not just using it — was what eventually allowed us to diagnose why transfer learning failed.

**Iterative masked decoding.** ESM3 is not an autoregressive model. Building a generation engine required implementing parallel iterative decoding: start fully masked, rank positions by model confidence, unmask the most confident first, repeat. Confidence-first unmasking ensures high-certainty anchors are placed early and provide context for subsequent positions.

### Masked Language Modeling

**Aligning training with inference.** A core challenge in MLM fine-tuning for generation is that the training distribution — partially masked sequences — differs from the inference condition — fully masked sequences. Masking strategy and loss design must account for this gap, not just optimize training metrics.

**Logarithmic loss boost.** We designed a custom loss weighting term that up-weights heavily masked samples: `weight = 1.0 + boost_ratio × log(N + 1)` where N counts masked positions across tracks. The logarithmic scaling reflects two properties of the task: short binders are the primary target so errors on longer sequences should contribute proportionally less, and masking 20 out of 40 positions is a genuinely harder problem than masking 2 out of 4 even at the same mask rate. The boost prepares the model for full-mask inference without over-penalizing edge cases at the long end of the length distribution.

### Fine-Tuning a Large Model in Production

End-to-end fine-tuning of a 1.4B parameter model on real hardware, from first principles. Not a tutorial, not a notebook — a production run on a Lightning AI H100 instance with a custom pipeline built from scratch.

**Learning rate and scheduling.** Tuned peak LR, warmup length, and cosine decay schedule empirically across two runs. Learned the hard way that too high a peak LR with insufficient warmup causes the model to overfit before the schedule has time to consolidate learning. The warmup phase is not cosmetic — it matters for which local minimum the model finds.

**Optimizer and memory.** Used 8-bit AdamW to fit optimizer states within GPU memory without sacrificing convergence. Combined with gradient checkpointing to trade compute for memory on activations. These two together made fine-tuning a 1.4B model on a single GPU feasible.

**Gradient accumulation and effective batch size.** Physical batch size was constrained by VRAM. Designed a crash test script to find the largest batch that fits, then computed gradient accumulation steps to reach a target effective batch size. Understood the difference between what the GPU sees and what the optimizer sees.

**Bucket batching.** Implemented a custom sampler that groups samples of similar sequence length into buckets, padding each batch only to the nearest multiple of 64. Avoids wasting compute on padding tokens and stabilizes training by keeping sequence lengths consistent within a batch.

**Flash Attention and torch.compile.** Enabled Flash Attention for memory-efficient attention computation. Navigated the incompatibility between `torch.compile` `reduce-overhead` mode and gradient checkpointing — CUDA graph capture conflicts with dynamic recomputation. Settled on `mode="default"` with `dynamic=True`.

**Checkpointing and resume.** Implemented checkpoint saving that stores both the LoRA adapter weights and the full training state — optimizer, scheduler, epoch. Identified and fixed a bug where best-model saves were missing `training_state.pt`, causing silent resume failures.

**Metrics and logging.** Logged per-epoch accuracy, perplexity, and raw loss split by sample type: Vinland sequence, PDB sequence, and PDB structure separately. This split was essential — aggregate loss hid the fact that the structure track was not learning while the sequence tracks improved. Perplexity is more interpretable than raw loss for generation tasks and allows meaningful comparison across vocabulary sizes.

---

## Why Transfer Learning Did Not Work

The inference test on a well-characterized target with structurally and functionally distinct binding domains produced indistinguishable outputs across domain windows. No motif discrimination. The failure is architectural and fundamental.

ESM3's geometric attention encodes **relative distances between residues within a chain**. The expectation was that two SH3 domains from different proteins — having similar relative distance patterns — would produce similar representations, allowing the model to cross-learn the association between SH3 geometry and proline-rich binding motifs.

The problem is that our fingerprints are **multi-domain**: a single chain contains surface fragments from multiple structural regions of the target protein, stitched together in their native spatial arrangement. Each such fingerprint is a geometrically unique object. The SH3 fragment embedded within the PIK3R1 fingerprint has a completely different spatial relationship to its neighboring residues than the SH3 fragment embedded within the ABL1 fingerprint. The model has no way to match these substructures across proteins. Rather than learning transferable domain grammar, it learned a rich per-protein barcode.

Transfer learning — the ability to apply learned binding rules to unseen proteins with structurally similar surfaces — was not achieved.

---

## What Would Be Required Next

The fix is a data pipeline problem, not a model architecture problem.

Each domain must be presented as an **isolated chain**. A standalone SH3 domain has the same relative distance signature as any other SH3 domain regardless of which protein it came from — the structural comparability ESM3 is capable of becomes accessible. But isolated domain chains require domain-resolved training pairs: you need to know which domain of the target each peptide actually binds, information that literature curation alone cannot provide at scale.

The correct path is to use a structure prediction tool such as AlphaFold Multimer or Boltz 2 to dock each peptide against its target, extract the contact interface, and use that as the training signal. Each (peptide, contact substructure) pair is an unambiguous, domain-resolved sample. The model would be forced to learn genuine structural grammar and cross-protein transfer would emerge naturally.

This requires significant compute for the docking step at Vinland scale, but produces a fundamentally correct training set. The multi-chain ESM3 architecture, the tokenizer, the training loop, and the iterative inference engine all carry forward unchanged.
