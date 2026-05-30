#! usr/bin/env python3

"""
Amino acid lookup tables and shared helper utilities.
GLOBAL CONSTANTS AND FUNCTIONS FOR FEATURE COMPUTATION AND TARGET CALCULATION.
"""

import numpy as np  # type: ignore[import-untyped, import-not-found]

# Amino acid vocabulary
AA20 = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
AA2IDX: dict[str, int] = {a: i for i, a in enumerate(AA20)}

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY: dict[str, float] = {
    "ALA": 1.8,
    "CYS": 2.5,
    "ASP": -3.5,
    "GLU": -3.5,
    "PHE": 2.8,
    "GLY": -0.4,
    "HIS": -3.2,
    "ILE": 4.5,
    "LYS": -3.9,
    "LEU": 3.8,
    "MET": 1.9,
    "ASN": -3.5,
    "PRO": -1.6,
    "GLN": -3.5,
    "ARG": -4.5,
    "SER": -0.8,
    "THR": -0.7,
    "VAL": 4.2,
    "TRP": -0.9,
    "TYR": -1.3,
}

AROMATIC: frozenset[str] = frozenset({"PHE", "TRP", "TYR"})
CHARGED: frozenset[str] = frozenset({"ARG", "LYS", "ASP", "GLU", "HIS"})
POLAR: frozenset[str] = frozenset({"SER", "THR", "ASN", "GLN", "TYR", "CYS"})
SMALL: frozenset[str] = frozenset({"GLY", "ALA", "SER", "CYS"})
BRANCHED: frozenset[str] = frozenset({"VAL", "LEU", "ILE"})

CHARGE_SIGN: dict[str, float] = {
    "ARG": +1,
    "LYS": +1,
    "ASP": -1,
    "GLU": -1,
    "HIS": +0.5,
}


# One-hot encoding
def one_hot_aa(resname: str) -> np.ndarray:
    """Return a length-20 one-hot vector for a canonical amino acid."""
    v = np.zeros(len(AA20), dtype=np.float32)
    if resname in AA2IDX:
        v[AA2IDX[resname]] = 1.0
    return v


# Druggability target computation
def compute_druggability_target(
    pocket: dict,
    adv_features: dict,
) -> float:
    """
    Heuristic druggability score ∈ [0, 1].

    Factors
    -------
    size_score: Gaussian centred at 15 residues (sweet-spot for a drug-sized binding site).
    volume_score: Gaussian centred at 500 Ang^3 (empirical drug-pocket volume).
    hydrophobicity_score: Hydrophobic contacts drive desolvation delG.
    compact_score: Higher compactness -> more enclosed cavity.
    charge_score: Mixed charge density (~0.3) aids H-bond networks.
    persistence: Cryptic pockets penalised (0.3x weight vs. 1.0).

    References
    ----------
    Halgren (2009) J. Chem. Inf. Model. 49, 377-389.
    Schmidtke & Barril (2010) J. Med. Chem. 53, 5858-5867.
    """
    size_score = float(np.exp(-((pocket["size"] - 15) ** 2) / 50))
    volume_score = float(
        np.exp(-((adv_features["convex_volume"] - 500) ** 2) / 100_000)
    )
    hydrophobicity_score = min(1.0, pocket["hydrophobicity"] / 2.0)
    compact_score = min(1.0, adv_features["compactness"] * 2)
    charge_score = float(np.exp(-((adv_features["charge_density"] - 0.3) ** 2) / 0.1))
    persistence = 1.0 if not pocket.get("is_cryptic", False) else 0.3

    score = (
        0.20 * size_score + 0.20 * volume_score + 0.20 * hydrophobicity_score + 0.15 * compact_score + 0.10 * charge_score + 0.15 * persistence
    )
    return float(np.clip(score, 0.0, 1.0))
