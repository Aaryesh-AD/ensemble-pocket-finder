# Pipeline: Extended Methodology

This document expands on the methodological choices in the original implementation of the notebook to the
`bioemu-pocket-discovery` pipeline.

---

## Why ensemble-based pocket detection?

Druggable pockets broadly fall into two categories:

1. **Persistent (orthosteric) pockets** which are present in the apo crystal structure or AlphaFold model.  Standard tools (FPocket, SiteMap) find these reliably.

2. **Cryptic pockets** only materialise in excited conformational states.
They are absent in X-ray or AlphaFold structures but can be revealed
by MD or much faster by using generative ensemble models such as BioEMU.
Approximately 20 % of known allosteric drug-binding sites fall into
this category (Cimermancic et al., 2016).

---

## BioEMU vs. MD sampling

| Property | Molecular Dynamics | BioEMU |
|---|---|---|
| Method | Newtonian physics integration | Energy-based generative model |
| Time for 200 conformations | ~10 hours (T4 GPU) | ~3-7 minute on avg|
| Thermodynamic accuracy | Ground truth (force-field limited) | Matches MD distributions on benchmarks |
| Coverage of rare states | Limited by simulation length | Improved: model trained on diverse structures |

BioEMU outputs are stored as a standard `topology.pdb` + `samples.xtc` pair,
making them drop-in compatible with MDAnalysis and any MD analysis workflow.

---

## Pocket detection: algorithm choices

### Why Cα atoms?

Using Cα (backbone) rather than all-atom coordinates:
- Dramatically reduces computational cost (1 atom per residue vs. ~7–15).
- Cα positions faithfully represent the residue's spatial location for
  coarse-grained pocket geometry.
- All physico-chemical features (hydrophobicity, charge, etc.) are computed
  per residue anyway, so all-atom positions provide no additional information
  at this stage.

### Adaptive clustering

Two regimes exist in our benchmark proteins:

**Small proteins (≤ 20 aa):** Entire protein can fit inside a single Ward
cluster distance threshold.  DBSCAN with a small `ε = 4–5 Å` avoids this
by requiring local density rather than global proximity.

**Larger proteins (> 20 aa):** Ward hierarchical clustering with threshold
`t = max(6 Å, 1.5 × R_g)` creates well-separated pocket clusters.

The `max_pocket_fraction` cap (default 0.6) prevents degenerate solutions
where the entire protein is labelled a single pocket.

---

## Graph featurisation: design decisions

### Why include global features?

Attention-based GNNs can sometimes be myopic, focussing on local edge
patterns while missing global pocket properties.  Passing an 11-dimensional
global descriptor directly to the fusion head bypasses this:

- **Convex hull volume/surface area** captures whether the pocket is a
  shallow groove or a deep enclosed cavity.
- **Compactness** κ = V/A: higher values indicate more enclosed cavities,
  which correlate with ligand binding affinity (Schmidtke & Barril, 2010).
- **Flexibility** (std of distances from centroid) proxies B-factor-like
  thermal motion within the pocket.

### Why multi-scale pooling?

Using mean, max, and sum pooling in parallel provides complementary views:

- **Mean pool** → normalised average residue character (invariant to size).
- **Max pool** → most chemically distinctive residue present.
- **Sum pool** → total capacity (scales with pocket size, relevant for
  estimating ligand size compatibility).

Concatenating all three before the fusion head lets the model learn which
combination is predictive, rather than committing to one.

---

## Training: handling class imbalance

The dataset is imbalanced: druggable (positive) pockets are a minority
because:
1. Many conformations of small proteins produce non-druggable pockets
   (too small or too flat).
2. Cryptic pockets are explicitly penalised in the target score.

Three complementary techniques address this:

1. **WeightedRandomSampler** ensures that each training batch contains
   approximately equal numbers of positive and negative graphs.
2. **Focal Loss** (γ = 2, α = 0.85) down-weights the loss from easy
   negatives, preventing gradient domination by the majority class.
3. **Threshold selection** (post-training, on calibrated probabilities)
   avoids the default 0.5 threshold which may be suboptimal under imbalance.

---

## GroupKFold: why conformer leakage matters

A naïve random train/test split would place different conformers of the
same protein in both sets.  This inflates performance because the model
sees near-identical graphs (same residue types, similar geometry) during
training and evaluation.

`GroupKFold` with groups derived from `conf_id // 10` ensures that
conformers from the same coarse temporal window of the BioEMU ensemble
stay in a single partition.  This measures genuine generalisation, not
memorisation of conformational states.

---

## Calibration: why Platt scaling?

GNN outputs are logits, not probabilities.  Without calibration:
- Threshold selection is arbitrary (the logit space has no natural scale).
- Probability estimates are unreliable for downstream use.

Platt scaling fits a logistic regression $\hat{p} = \sigma(a \cdot z + b)$
on validation logits, mapping them to calibrated probabilities.  It is a
1D problem (two parameters) that generalises well even on small validation sets.

---

## Limitations and future directions

- **No 3D structure → pocket volume** mapping: `volume = size × 30 Å³`
  is a rough approximation.  Full all-atom pocket volumes via probe-based
  methods (POVME, fpocket) would improve target quality.
- **Target score is heuristic**: training labels come from a hand-crafted
  score, not experimental binding data.  Future work could use DUD-E or
  PocketMiner-validated labels.
- **Small protein artefacts**: miniproteins (chignolin, 10 aa) produce very
  few residues per pocket, making graph structure degenerate.  A surface
  accessibility filter (SASA) would help.
- **Attention visualisation**: `model.attention_weights()` exposes GAT-1
  attention for interpretability analysis not yet connected to a notebook
  visualisation.


> [!Note]
> This project is just a proof-of-concept for the potential of BioEMU in drug discovery.  The pipeline is not production-ready and should be viewed as a testbed only. The methodological choices were made for simplicity and demonstration purposes, not as a definitive solution to the complex problem of pocket detection.
