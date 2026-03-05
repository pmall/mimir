# MÍMIR

**MÍMIR** is a generative biology framework designed to design _de novo_ peptide binders for specific human proteins.

## The Concept

Nature has already designed highly optimized peptide binders: viruses, over millions of years of coevolution, have evolved short sequences that bind human proteins with extreme specificity to hijack cellular machinery or evade immune responses.

MÍMIR is built upon this biological foundation. Utilizing the **Vinland** database—a massive manual curation of experimentally supported protein-protein and host-virus interactions—MÍMIR trains directly on the validated, functional sequences of biology.

However, evolution has only explored a tiny fraction of possible sequence space. MÍMIR learns the underlying physical grammar of these interactions and applies it to propose structurally compatible binders that evolution never produced. This allows the framework to zero in on targets that may have no known natural binder at all.

## Core Architecture: MÍMIR v2

MÍMIR v2 represents a structural leap in generative binder design. Rather than treating target proteins as simple textual identifiers or linear 1D sequences, MÍMIR explicitly models the physical, 3D interface where binding occurs.

### 1. The Structural Fingerprint

Proteins often contain hundreds or thousands of amino acids, but binding is mediated by a highly specific, accessible surface. MÍMIR v2 utilizes a dynamic **fingerprinting protocol** to compress high-resolution structural data (sourced from the human proteome) into a dense, biologically relevant representation. It applies strict physical gates:

- **Rigidity Gate:** Discards disordered loops or highly flexible regions that do not provide a stable docking surface for a peptide.
- **Surface Gate:** Iteratively trims buried, inaccessible residues while strictly preserving concave cavities and binding pockets—the typical anchor points for high-affinity interactions.

The model is conditioned explicitly on this structural surface, preserving the actual spatial relationships and distances between the critical residues.

### 2. Generalization by Structural Similarity

A fundamental requirement for _de novo_ design is the ability to target proteins the model has never seen. MÍMIR v2 achieves this by generalizing at the structural domain level rather than the target protein level.

When presented with the structural fingerprint of a novel protein, MÍMIR inherently recognizes its individual structural domains. By aggregating learned binding constraints from multiple training proteins that share those same or similar domains, the model accurately interpolates binding knowledge across the broad domain space.

### 3. Asymmetric Supervision

MÍMIR v2 is built by fine-tuning **ESM3 (1.4B)**, utilizing its sequence, structure, and SASA (Solvent Accessible Surface Area) tracks. The training process uses a multi-task regimen that balances two distinct data types:

- PDB-derived complexes that provide exact 3D structural ground truth for the binder geometry.
- A massive dataset of sequence-only binders curated from literature, lacking explicit structural measurements.

By employing an asymmetric masking strategy, MÍMIR extracts maximum value from both. The structural complexes teach the precise physics of docking to a target surface, while the large sequence-only dataset teaches the immense genetic diversity of valid binding motifs.

### 4. Contextual Structure Inference

Because MÍMIR learns the direct relationship between target geometry and binder sequence, it possesses the emergent capability of bound-state structure inference. Given an arbitrary binder sequence for a known target protein, MÍMIR can predict the 3D structure of the binder _specifically in the context of its binding partner_, providing a functionally accurate conformation rather than attempting to fold a short peptide in a vacuum.

## Technical Setup

- **Foundation Engine:** ESM3 1.4B, fine-tuned efficiently via LoRA to handle massive vocabulary constraints.
- **Spatial Anchoring:** Target interactions are separated via a custom positional encoding strategy (`<cut>` token boundaries and absolute RoPE position ID jumps). This explicitly informs the model it is processing two disjoint molecules interacting in physical space, without requiring foundational architectural modifications.
- **Generation Strategy:** Generates novel peptide sequences via Parallel Iterative Decoding through a Masked Language Modeling objective, gradually "sculpting" structurally valid sequences out of pure noise, conditioned entirely on the target's physical surface.
