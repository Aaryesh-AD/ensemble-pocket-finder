"""
Pockets using PyTorch Geometric graph conversion.

PocketGraphBuilder
------------------
Converts a list of pocket dicts (from EnsemblePocketFinder) into
`torch_geometric.data.Data` objects for GNN training.

Graph schema
~~~~~~~~~~~~
Nodes: one per residue in the pocket
          Feature vector (30 dims):
            [0:20]: one-hot amino-acid type
            [20]: Kyte-Doolittle hydrophobicity (normalised ÷ 5)
            [21]: aromatic flag
            [22]: polar flag
            [23]: charged flag
            [24]: small flag
            [25]: branched flag
            [26]: relative burial depth (depth / max_depth)
            [27]: relative neighbour count (n / max_n)
            [28]: relative distance to pocket centroid
            [29]: log(1 + neighbour_count)

Edges: all residue pairs within 10 Angstrom cutoff (bidirectional)
          Attribute vector (4 dims):
            [0]: distance kernel  1/(1+d)
            [1]: hydrophobic potential  h_i x h_j / 25
            [2]: electrostatic sign  -q_i x q_j
            [3]: aromatic-aromatic flag

Global: 11-dim pocket-level descriptor (size, convex volume, surface area, compactness, charge density, helix/sheet propensity, flexibility, hydrophobicity, aromatic content, std depth)

Target: druggability score in [0,1] from compute_druggability_target(), binarised at 0.5 before training.
"""

import numpy as np  # type: ignore[import-untyped, import-not-found]
import torch  # type: ignore[import-untyped, import-not-found]
from scipy.spatial.distance import pdist  # type: ignore[import-untyped, import-not-found]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped, import-not-found]
from torch_geometric.data import Data  # type: ignore[import-untyped, import-not-found]

from bioemu_pocket.utils import (  # type: ignore[import-untyped]
    AROMATIC,
    BRANCHED,
    CHARGE_SIGN,
    CHARGED,
    HYDROPHOBICITY,
    POLAR,
    SMALL,
    compute_druggability_target,
    one_hot_aa,
)


class PocketGraphBuilder:
    """WE Convert detected pockets into PyG Data objects."""

    def __init__(self, pocket_data: list[dict], persistence_cutoff: float = 0.3):
        self.pocket_data = pocket_data
        self.persistence_cutoff = persistence_cutoff
        self.scaler = StandardScaler()

    # Advanced per-pocket geometric features
    @staticmethod
    def _advanced_features(pocket: dict) -> dict:
        residues = pocket["residues"]
        coords = np.array([r["coord"] for r in residues])
        centroid = coords.mean(axis=0)

        if len(coords) >= 4:
            try:
                from scipy.spatial import ConvexHull  # type: ignore[import-untyped]

                hull = ConvexHull(coords)
                convex_vol = hull.volume
                surface_area = hull.area
            except Exception:
                convex_vol = pocket["volume"]
                surface_area = pocket["size"] * 10.0
        else:
            convex_vol = pocket["volume"]
            surface_area = pocket["size"] * 10.0

        compactness = convex_vol / (surface_area + 1e-6)
        charge_count = sum(1 for r in residues if r["residue_name"] in CHARGED)
        charge_density = charge_count / (pocket["size"] + 1e-6)

        helix_formers = {"ALA", "GLU", "LEU", "MET"}
        sheet_formers = {"VAL", "ILE", "TYR", "TRP", "PHE"}
        helix_prop = (
            sum(1 for r in residues if r["residue_name"] in helix_formers)
            / pocket["size"]
        )
        sheet_prop = (
            sum(1 for r in residues if r["residue_name"] in sheet_formers)
            / pocket["size"]
        )

        dists_from_centroid = np.linalg.norm(coords - centroid, axis=1)
        flexibility = np.std(dists_from_centroid) / (
            np.mean(dists_from_centroid) + 1e-6
        )

        return {
            "convex_volume": convex_vol,
            "surface_area": surface_area,
            "compactness": compactness,
            "charge_density": charge_density,
            "helix_propensity": helix_prop,
            "sheet_propensity": sheet_prop,
            "flexibility": flexibility,
            "std_depth": float(np.std([r["depth"] for r in residues])),
        }

    # Node feature construction (30-dim)
    @staticmethod
    def _node_features(residue: dict, pocket_stats: dict) -> np.ndarray:
        rn = residue["residue_name"]
        coord = residue["coord"]

        aa_feats = np.array(
            [
                HYDROPHOBICITY.get(rn, 0.0) / 5.0,
                1.0 if rn in AROMATIC else 0.0,
                1.0 if rn in POLAR else 0.0,
                1.0 if rn in CHARGED else 0.0,
                1.0 if rn in SMALL else 0.0,
                1.0 if rn in BRANCHED else 0.0,
            ],
            dtype=np.float32,
        )

        rel_depth = residue["depth"] / (pocket_stats["max_depth"] + 1e-6)
        rel_neigh = residue["neighbors"] / (pocket_stats["max_neighbors"] + 1e-6)
        dist_to_center = np.linalg.norm(coord - pocket_stats["center"])
        rel_dist = dist_to_center / (pocket_stats["max_dist"] + 1e-6)
        log_neigh = np.log1p(residue["neighbors"])

        return np.concatenate(
            [
                one_hot_aa(rn),  # DIM: 20
                aa_feats,  # DIM: 6
                [rel_depth, rel_neigh, rel_dist, float(log_neigh)],  # 4
            ]
        ).astype(np.float32)  # total: 30

    # Edge feature construction (4-dim)
    @staticmethod
    def _edge_features(
        coord_i: np.ndarray,
        coord_j: np.ndarray,
        res_i: dict,
        res_j: dict,
    ) -> torch.Tensor:
        dist = float(np.linalg.norm(coord_i - coord_j))
        h_i = HYDROPHOBICITY.get(res_i["residue_name"], 0.0)
        h_j = HYDROPHOBICITY.get(res_j["residue_name"], 0.0)
        q_i = CHARGE_SIGN.get(res_i["residue_name"], 0.0)
        q_j = CHARGE_SIGN.get(res_j["residue_name"], 0.0)
        arom = float(
            res_i["residue_name"] in AROMATIC and res_j["residue_name"] in AROMATIC
        )
        return torch.tensor(
            [1.0 / (1.0 + dist), (h_i * h_j) / 25.0, -q_i * q_j, arom],
            dtype=torch.float32,
        )

    # Main graph object builder
    def build_graph(self, pocket: dict) -> Data | None:
        """Convert a single pocket dict to PyG Data object."""
        residues = pocket["residues"]
        n = len(residues)
        if n < 3:
            return None

        coords = np.array([r["coord"] for r in residues])
        pocket_stats = {
            "center": coords.mean(axis=0),
            "max_depth": max(r["depth"] for r in residues),
            "max_neighbors": max(r["neighbors"] for r in residues),
            "max_dist": np.max(pdist(coords)) if n > 1 else 1.0,
        }

        # Node features
        x = torch.tensor(
            np.stack([self._node_features(r, pocket_stats) for r in residues]),
            dtype=torch.float32,
        )

        # Edges
        edge_index, edge_attr = [], []
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(coords[i] - coords[j]))
                if d < 10.0:
                    ef = self._edge_features(
                        coords[i], coords[j], residues[i], residues[j]
                    )
                    edge_index += [[i, j], [j, i]]
                    edge_attr += [ef, ef]

        if not edge_index:
            # Fallback using: k=3 nearest neighbours
            for i in range(n):
                dists = [np.linalg.norm(coords[i] - coords[j]) for j in range(n)]
                for j in np.argsort(dists)[1 : min(4, n)]:
                    ef = self._edge_features(
                        coords[i], coords[j], residues[i], residues[j]
                    )
                    edge_index += [[i, j], [j, i]]
                    edge_attr += [ef, ef]

        edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_t = torch.stack(edge_attr) if edge_attr else torch.empty((0, 4))

        adv = self._advanced_features(pocket)
        u = torch.tensor(
            [
                pocket["size"] / 20.0,
                adv["convex_volume"] / 1000.0,
                adv["surface_area"] / 500.0,
                adv["compactness"],
                adv["charge_density"],
                adv["helix_propensity"],
                adv["sheet_propensity"],
                adv["flexibility"],
                pocket["hydrophobicity"],
                float(pocket["has_aromatic"]),
                adv["std_depth"] / 10.0,
            ],
            dtype=torch.float32,
        )

        target = compute_druggability_target(pocket, adv)
        return Data(
            x=x,
            edge_index=edge_index_t,
            edge_attr=edge_attr_t,
            y=torch.tensor([target], dtype=torch.float32),
            u=u,
            pos=torch.tensor(coords, dtype=torch.float32),
            num_nodes=n,
        )

    def build_all(self) -> list[Data]:
        """Build graphs for all pockets with size ≥ min_residues (3)."""
        graphs = []
        for pocket in self.pocket_data:
            if pocket["size"] >= 3:
                g = self.build_graph(pocket)
                if g is not None:
                    graphs.append(g)
        print(f"Done Building {len(graphs)} pocket graphs")
        if graphs:
            print(f"Node features : {graphs[0].x.shape[1]} dims")
            print(f"Edge features : {graphs[0].edge_attr.shape[1]} dims")
            print(f"Global features: {graphs[0].u.shape[0]} dims")
        return graphs
