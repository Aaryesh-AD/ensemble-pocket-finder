"""
Ensemble-based pocket detection from BioEMU conformational trajectories.

EnsemblePocketFinder
--------------------
Operates on a topology (.pdb) + trajectory (.xtc) pair produced by BioEMU,
iterating over frames and clustering C-alpha atoms by spatial proximity to find
candidate binding sites.

Pocket classification (persistent vs. cryptic)
----------------------------------------------
A pocket is persistent when its centroid appears in ≥ `cryptic_frac` of frames (default 30 %).
All others are labelled cryptic transient sites that open and close on the conformational ensemble.

This mirrors the definition used in:
  Cimermancic et al. (2016) J. Mol. Biol. 428, 709-719.
  Vajda et al. (2018) J. Med. Chem. 61, 2965-2981.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np  # type: ignore[import-untyped, import-not-found]
from scipy.cluster.hierarchy import fcluster, linkage   # type: ignore[import-untyped, import-not-found]
from scipy.spatial.distance import cdist    # type: ignore[import-untyped, import-not-found]
from sklearn.cluster import DBSCAN, KMeans      # type: ignore[import-untyped, import-not-found]

import MDAnalysis as mda     # type: ignore[import-untyped, import-not-found]

from bioemu_pocket.utils import HYDROPHOBICITY, AROMATIC        # type: ignore[import-untyped, import-not-found]

warnings.filterwarnings("ignore")


# EnsemblePocketFinder
@dataclass
class EnsemblePocketFinder:
    """
    Detect and classify druggable pockets across a conformational ensemble.

    Parameters
    ----------
    top_path : str
        Path to the PDB topology file (first frame / reference structure).
    xtc_path : str
        Path to the XTC trajectory file produced by BioEMU.
    min_residues : int
        Minimum residues required to call a cluster a pocket.
    distance_cutoff : float | None
        Å cutoff for residue-residue contact; None → adaptive per frame.
    bin_step : float
        Spatial bin size (Å) used in the spatial-hash persistence check.
    cryptic_frac : float
        Fraction-of-frames threshold: pockets seen in fewer frames are
        labelled cryptic.
    max_pocket_fraction : float
        Upper cap on pocket size as a fraction of total protein length.
        Prevents the whole protein being flagged as a single pocket.
    """

    top_path: str
    xtc_path: str
    min_residues: int = 3
    distance_cutoff: float | None = 8.0
    bin_step: float = 0.5
    cryptic_frac: float = 0.30
    max_pocket_fraction: float = 0.60

    # Set by _setup()
    u: mda.Universe = field(init=False, repr=False)
    ca: mda.AtomGroup = field(init=False, repr=False)
    resnames: np.ndarray = field(init=False, repr=False)
    resids: np.ndarray = field(init=False, repr=False)
    n_frames: int = field(init=False, repr=False)

    def _setup(self) -> None:
        self.u = mda.Universe(self.top_path, self.xtc_path)
        self.ca = self.u.select_atoms("protein and name CA")
        self.resnames = np.array([r.resname for r in self.ca.residues])
        self.resids = np.array([r.resid for r in self.ca.residues])
        self.n_frames = len(self.u.trajectory)

    def _cluster(
        self,
        coords: np.ndarray,
        candidates: list[dict],
        rg: float,
    ) -> list[dict]:
        """Group candidate residues into pocket clusters."""
        if len(candidates) < self.min_residues:
            return []

        X = np.array([c["coord"] for c in candidates], dtype=float)
        n_protein = len(self.ca)
        max_pocket_size = int(n_protein * self.max_pocket_fraction)

        if n_protein < 20:
            clustering = DBSCAN(eps=4.0 if n_protein <= 10 else 5.0, min_samples=2).fit(
                X
            )
            labels = clustering.labels_
        else:
            Z = linkage(X, method="ward")
            t = max(6.0, 1.5 * rg)
            labels = fcluster(Z, t=t, criterion="distance")

        pockets: list[dict] = []
        for cid in np.unique(labels):
            if cid == -1:  # DBSCAN noise
                continue
            grp = [candidates[i] for i in range(len(candidates)) if labels[i] == cid]

            if len(grp) > max_pocket_size:
                if n_protein <= 20:
                    grp_coords = np.array([g["coord"] for g in grp])
                    n_sub = max(2, len(grp) // 5)
                    sub_labels = KMeans(
                        n_clusters=n_sub, random_state=16, n_init=10
                    ).fit_predict(grp_coords)
                    for sub_id in range(n_sub):
                        sub = [
                            grp[i] for i in range(len(grp)) if sub_labels[i] == sub_id
                        ]
                        if self.min_residues <= len(sub) <= max_pocket_size:
                            pockets.append(self._pocket_dict(sub))
                continue

            if len(grp) >= self.min_residues:
                pockets.append(self._pocket_dict(grp))

        if n_protein <= 15:
            pockets = pockets[:2]
        elif n_protein <= 30:
            pockets = pockets[:3]
        return pockets

    @staticmethod
    def _pocket_dict(grp: list[dict]) -> dict:
        return {
            "residues": grp,
            "size": len(grp),
            "center": np.mean([g["coord"] for g in grp], axis=0),
        }

    def _detect_pockets_from_coords(self, coords: np.ndarray) -> list[dict]:
        """Main per-frame pocket detection logic."""
        n = coords.shape[0]
        if n < self.min_residues:
            return []

        center = coords.mean(axis=0)
        dmat = cdist(coords, coords)
        rg = float(np.sqrt(((coords - center) ** 2).sum() / n + 1e-9))

        # ---- adaptive parameters by protein size ---- #
        if n <= 10:
            return self._tiny_protein_pockets(coords, dmat, center, rg)

        if n <= 20:
            cutoff = self.distance_cutoff or 8.0
            neigh_lo, neigh_hi = 1, min(8, n - 2)
            depth_lo, depth_hi = 0.0, 3.0 * rg
        elif n <= 40:
            cutoff = max(6.0, self.distance_cutoff or 8.0)
            neigh_lo = 2
            neigh_hi = max(neigh_lo + 1, min(12, int(0.6 * n)))
            depth_lo, depth_hi = 0.1 * rg, 2.8 * rg
        else:
            cutoff = self.distance_cutoff or max(5.0, 0.8 * max(rg, 8.0))
            neigh_lo = max(2, int(0.1 * n))
            neigh_hi = max(neigh_lo + 1, int(0.7 * n))
            depth_lo, depth_hi = 0.2 * rg, 2.5 * rg

        cand: list[dict] = []
        for i in range(n):
            neighbors = int((dmat[i] < cutoff).sum() - 1)
            depth = float(np.linalg.norm(coords[i] - center))
            if neigh_lo <= neighbors <= neigh_hi and depth_lo <= depth <= depth_hi:
                cand.append(self._res_dict(i, coords[i], neighbors, depth))

        if n <= 20 and len(cand) < n * 0.3:
            for i in range(n):
                if not any(c["residue_id"] == self.resids[i] for c in cand):
                    neighbors = int((dmat[i] < cutoff).sum() - 1)
                    if neighbors >= 2:
                        cand.append(
                            self._res_dict(
                                i,
                                coords[i],
                                neighbors,
                                float(np.linalg.norm(coords[i] - center)),
                            )
                        )

        if len(cand) < self.min_residues:
            return []
        return self._cluster(coords, cand, rg)

    def _tiny_protein_pockets(
        self, coords: np.ndarray, dmat: np.ndarray, center: np.ndarray, rg: float
    ) -> list[dict]:
        """Special-case detection for proteins ≤ 10 residues."""
        try:
            from scipy.spatial import ConvexHull        # type: ignore[import-untyped, import-not-found]

            hull = ConvexHull(coords)
            hull_pts = set(hull.vertices)
            non_hull = [i for i in range(len(coords)) if i not in hull_pts]
            if len(non_hull) >= 2:
                cand = [
                    self._res_dict(
                        i,
                        coords[i],
                        int((dmat[i] < 5.0).sum()) - 1,
                        float(np.linalg.norm(coords[i] - center)),
                    )
                    for i in non_hull
                ]
                for i in hull_pts:
                    if any(dmat[i, j] < 5.0 for j in non_hull):
                        cand.append(
                            self._res_dict(
                                i,
                                coords[i],
                                int((dmat[i] < 5.0).sum()) - 1,
                                float(np.linalg.norm(coords[i] - center)),
                            )
                        )
                if len(cand) >= self.min_residues:
                    return self._cluster(coords, cand, rg)
        except Exception:
            pass

        densities = [(dmat[i] < 5.0).sum() for i in range(len(coords))]
        top = np.argsort(densities)[-6:]
        cand = [
            self._res_dict(
                i,
                coords[i],
                densities[i] - 1,
                float(np.linalg.norm(coords[i] - center)),
            )
            for i in top
        ]
        return self._cluster(coords, cand, rg)

    def _res_dict(
        self, idx: int, coord: np.ndarray, neighbors: int, depth: float
    ) -> dict:
        return {
            "residue_id": int(self.resids[idx]),
            "residue_name": str(self.resnames[idx]),
            "coord": coord,
            "neighbors": neighbors,
            "depth": depth,
        }

    # Public API
    def run(self, max_frames: int | None = None) -> tuple[list[dict], list[int]]:
        """
        Run pocket detection over the trajectory.

        Returns
        -------
        pocket_data: list[dict]
            One entry per detected pocket per frame, with geometry and
            physicochemical annotations.
        pockets_per_conf: list[int]
            Number of pockets found in each frame.
        """
        self._setup()
        limit = min(max_frames or self.n_frames, self.n_frames)
        pocket_data: list[dict] = []
        pockets_per_conf: list[int] = []

        print(f"Processing {limit} frames | protein length: {len(self.ca)} residues")

        for fi, _ in enumerate(self.u.trajectory[:limit]):
            coords = self.ca.positions.copy()
            pockets = self._detect_pockets_from_coords(coords)
            pockets_per_conf.append(len(pockets))

            for pk in pockets:
                hydros = [
                    HYDROPHOBICITY.get(r["residue_name"], 0.0) for r in pk["residues"]
                ]
                arom = sum(1 for r in pk["residues"] if r["residue_name"] in AROMATIC)
                pocket_data.append(
                    {
                        "conf_id": fi,
                        "size": pk["size"],
                        "hydrophobicity": float(np.mean(hydros)) / 5.0
                        if hydros
                        else 0.0,
                        "has_aromatic": int(arom > 0),
                        "depth": float(np.mean([r["depth"] for r in pk["residues"]])),
                        "volume": pk["size"] * 30.0,
                        "center": pk["center"],
                        "residues": pk["residues"],
                    }
                )

            if (fi + 1) % 50 == 0:
                print(
                    f"  {fi + 1}/{limit} frames processed | {len(pocket_data)} pockets so far"
                )

        print(f"\nTotal pockets: {len(pocket_data)}")
        if pocket_data:
            sizes = [p["size"] for p in pocket_data]
            print(f"Size range: {min(sizes)}–{max(sizes)} residues")
        return pocket_data, pockets_per_conf

    def mark_cryptic(self, pocket_data: list[dict], n_frames_used: int) -> None:
        """
        Label each pocket as persistent or cryptic in-place.

        A pocket centroid is hashed into a spatial bin of side `bin_step` Ang.
        Bins that appear in ≥ `cryptic_frac x n_frames_used` frames are
        considered persistent.  When small proteins produce all-cryptic results
        (a degeneracy artefact of sparse sampling), the three most-frequent spatial bins are promoted to persistent.
        """
        if not pocket_data:
            return

        n = len(self.ca)
        bin_step = 3.0 if n <= 30 else self.bin_step
        thresh_frac = 0.15 if n <= 30 else self.cryptic_frac
        thresh = thresh_frac * n_frames_used

        def bin_key(x: np.ndarray) -> tuple:
            return tuple((np.asarray(x) / bin_step).round().astype(int))

        counts: dict[tuple, int] = {}
        for p in pocket_data:
            k = bin_key(p["center"])
            counts[k] = counts.get(k, 0) + 1

        # Secondary check via DBSCAN within same-size groups
        size_groups: dict[int, list[tuple[int, dict]]] = {}
        for i, p in enumerate(pocket_data):
            size_groups.setdefault(p["size"], []).append((i, p))

        persistent_indices: set[int] = set()
        for size, group in size_groups.items():
            if len(group) < thresh:
                continue
            centers = np.array([p["center"] for _, p in group])
            labels = (
                DBSCAN(eps=5.0, min_samples=max(1, int(thresh / 2)))
                .fit(centers)
                .labels_
            )
            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue
                members = [
                    group[i][0] for i in range(len(group)) if labels[i] == cluster_id
                ]
                if len(members) >= thresh:
                    persistent_indices.update(members)

        for i, p in enumerate(pocket_data):
            p["is_cryptic"] = not (
                counts[bin_key(p["center"])] >= thresh or i in persistent_indices
            )

        # Fallback: ensure some pockets are persistent
        if all(p["is_cryptic"] for p in pocket_data):
            top_keys = {k for k, _ in sorted(counts.items(), key=lambda x: -x[1])[:3]}
            for p in pocket_data:
                if bin_key(p["center"]) in top_keys:
                    p["is_cryptic"] = False

        n_c = sum(p["is_cryptic"] for p in pocket_data)
        n_p = len(pocket_data) - n_c
        print(
            f"Persistent: {n_p} ({100 * n_p / len(pocket_data):.1f}%)  "
            f"Cryptic: {n_c} ({100 * n_c / len(pocket_data):.1f}%)"
        )
