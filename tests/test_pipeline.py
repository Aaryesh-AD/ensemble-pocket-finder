"""
JUST CONCEPT: Lightweight unit tests for the bioemu_pocket pipeline.

These tests run without BioEMU or actual trajectory files they
synthesise minimal fake data to verify the shape and contract of
each pipeline stage.

Run with:
    uv run pytest tests/ -v
"""

import numpy as np      # type: ignore[import-not-found, import-untyped]
import pytest       # type: ignore[import-not-found, import-untyped]
from torch_geometric.data import Batch      # type: ignore[import-not-found, import-untyped]

from bioemu_pocket.graph_builder import PocketGraphBuilder      # type: ignore[import-not-found, import-untyped]
from bioemu_pocket.model import AttentivePocketGNN, FocalLoss   # type: ignore[import-not-found, import-untyped]
from bioemu_pocket.trainer import choose_threshold  # type: ignore[import-not-found, import-untyped]
from bioemu_pocket.utils import one_hot_aa, compute_druggability_target     # type: ignore[import-not-found, import-untyped]


# Fixtures
def _fake_pocket(n_residues: int = 8, is_cryptic: bool = False) -> dict:
    """Return a pocket dict with synthetic C-alpha coordinates."""
    rng = np.random.default_rng(16)     # lucky number
    residues = []
    coords = rng.uniform(-10, 10, (n_residues, 3))
    for i, coord in enumerate(coords):
        residues.append(
            {
                "residue_id": i + 1,
                "residue_name": "ALA",
                "coord": coord,
                "neighbors": 4,
                "depth": float(np.linalg.norm(coord)),
            }
        )
    return {
        "conf_id": 0,
        "size": n_residues,
        "hydrophobicity": 0.36,
        "has_aromatic": 0,
        "depth": 5.0,
        "volume": n_residues * 30.0,
        "center": coords.mean(axis=0),
        "residues": residues,
        "is_cryptic": is_cryptic,
    }


class TestUtils:
    def test_one_hot_known(self):
        v = one_hot_aa("ALA")
        assert v.shape == (20,)
        assert v[0] == 1.0 and v.sum() == 1.0

    def test_one_hot_unknown(self):
        v = one_hot_aa("UNK")
        assert v.sum() == 0.0

    def test_druggability_target_range(self):
        pocket = _fake_pocket(12)
        adv = {
            "convex_volume": 400.0,
            "surface_area": 200.0,
            "compactness": 2.0,
            "charge_density": 0.3,
            "helix_propensity": 0.2,
            "sheet_propensity": 0.2,
            "flexibility": 0.1,
            "std_depth": 1.0,
        }
        score = compute_druggability_target(pocket, adv)
        assert 0.0 <= score <= 1.0

    def test_cryptic_penalty(self):
        pocket_p = _fake_pocket(12, is_cryptic=False)
        pocket_c = _fake_pocket(12, is_cryptic=True)
        adv = {
            "convex_volume": 500.0, "surface_area": 300.0, "compactness": 1.67,
            "charge_density": 0.3, "helix_propensity": 0.2, "sheet_propensity": 0.2,
            "flexibility": 0.1, "std_depth": 1.0,
        }
        sp = compute_druggability_target(pocket_p, adv)
        sc = compute_druggability_target(pocket_c, adv)
        assert sp > sc, "Persistent pockets should score higher than cryptic ones"


# POCKET GRAPH BUILDER
class TestGraphBuilder:
    def test_build_single_graph(self):
        pocket = _fake_pocket(8)
        builder = PocketGraphBuilder([pocket])
        g = builder.build_graph(pocket)
        assert g is not None
        assert g.x.shape == (8, 30)
        assert g.edge_attr.shape[1] == 4
        assert g.u.shape == (11,)
        assert 0.0 <= g.y.item() <= 1.0

    def test_too_small_pocket_returns_none(self):
        pocket = _fake_pocket(2)
        builder = PocketGraphBuilder([pocket])
        g = builder.build_graph(pocket)
        assert g is None

    def test_build_all(self):
        pockets = [_fake_pocket(n) for n in [4, 6, 8, 10]]
        builder = PocketGraphBuilder(pockets)
        graphs = builder.build_all()
        assert len(graphs) == 4


# AttentivePocketGNN
class TestModel:
    @pytest.fixture
    def graphs(self):
        pockets = [_fake_pocket(n) for n in [4, 6, 8, 10, 5, 7]]
        return PocketGraphBuilder(pockets).build_all()

    def test_forward_shape(self, graphs):
        g = graphs[0]
        model = AttentivePocketGNN(
            node_dim=g.x.shape[1],
            edge_dim=g.edge_attr.shape[1],
            global_dim=g.u.shape[0],
            hidden_dim=32,
            heads=4,
        )
        batch = Batch.from_data_list([g])
        out = model(batch)
        assert out.shape == (1,)

    def test_batch_forward(self, graphs):
        g = graphs[0]
        model = AttentivePocketGNN(
            node_dim=g.x.shape[1],
            edge_dim=g.edge_attr.shape[1],
            global_dim=g.u.shape[0],
            hidden_dim=32,
            heads=4,
        )
        batch = Batch.from_data_list(graphs[:4])
        out = model(batch)
        assert out.shape == (4,)

    def test_focal_loss_scalar(self, graphs):
        g = graphs[0]
        model = AttentivePocketGNN(
            node_dim=g.x.shape[1], edge_dim=g.edge_attr.shape[1],
            global_dim=g.u.shape[0], hidden_dim=32, heads=4,
        )
        loss_fn = FocalLoss()
        batch = Batch.from_data_list(graphs[:4])
        logits = model(batch)
        targets = batch.y.view(-1).float()
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0.0

    def test_hidden_dim_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            AttentivePocketGNN(node_dim=30, edge_dim=4, global_dim=11,
                               hidden_dim=33, heads=4)


# Threshold selection
class TestThreshold:
    def test_returns_float(self):
        rng = np.random.default_rng(0)
        y = (rng.random(100) > 0.5).astype(int)
        p = rng.random(100)
        thr = choose_threshold(y, p)
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0

    def test_high_precision_constraint(self):
        # If all positives are easy, threshold should be low and precision high
        y = np.array([0] * 80 + [1] * 20)
        p = np.array([0.1] * 80 + [0.9] * 20)
        thr = choose_threshold(y, p, min_precision=0.90)
        preds = (p >= thr).astype(int)
        tp = ((y == 1) & (preds == 1)).sum()
        fp = ((y == 0) & (preds == 1)).sum()
        precision = tp / max(tp + fp, 1e-9)
        assert precision >= 0.90
