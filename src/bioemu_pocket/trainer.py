"""
Training, evaluation, and calibration utilities.

Trainer
-------
Full training loop with:
  - AdamW optimizer + ReduceLROnPlateau scheduler
  - Gradient clipping (max-norm = 1.0)
  - AUROC + AUPRC validation tracking
  - Early stopping on AUPRC
  - Platt (logistic regression) post-hoc calibration
  - Threshold selection maximising F1 s.t. precision ≥ min_precision

Splits
------
build_group_splits() performs a 5-fold GroupKFold split keeping all
conformers of the same protein in a single partition, preventing data
leakage between conformational states of the same sequence.
"""

import copy

import numpy as np      # type: ignore[import-untyped, import-not-found]
import torch    # type: ignore[import-untyped, import-not-found]
import torch.nn as nn   # type: ignore[import-untyped, import-not-found]
from sklearn.linear_model import LogisticRegression     # type: ignore[import-untyped, import-not-found]
from sklearn.metrics import (   # type: ignore[import-untyped, import-not-found]
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold      # type: ignore[import-untyped, import-not-found]
from torch_geometric.data import Data       # type: ignore[import-untyped, import-not-found]
from torch_geometric.loader import DataLoader       # type: ignore[import-untyped, import-not-found]

from bioemu_pocket.model import FocalLoss       # type: ignore[import-untyped, import-not-found]


# Threshold selection
def choose_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.50,
    guard: tuple[float, float] = (0.05, 0.95),
) -> float:
    """
    Select the classification threshold that maximises F1 subject to
    precision >= ``min_precision``.  Falls back to Youden's J if no
    threshold satisfies the precision constraint.

    Parameters
    ----------
    y_true: array-like, shape (N,)
    y_prob: array-like, shape (N,)
    min_precision: float
    guard: (lo, hi) clips the candidate threshold range.

    Returns
    -------
    float  optimal threshold
    """
    probs = np.asarray(y_prob)
    y = np.asarray(y_true).astype(int)

    candidates = np.clip(np.unique(probs), *guard)
    candidates = np.unique(np.r_[guard[0], candidates, guard[1]])

    best_f1: dict = {"thr": 0.5, "f1": -1.0}
    best_yj: dict = {"thr": 0.5, "youden": -1.0}

    for t in candidates:
        y_pred = (probs >= t).astype(int)
        tp = ((y == 1) & (y_pred == 1)).sum()
        tn = ((y == 0) & (y_pred == 0)).sum()
        fp = ((y == 0) & (y_pred == 1)).sum()
        fn = ((y == 1) & (y_pred == 0)).sum()
        eps = 1e-12
        prec = tp / max(tp + fp, eps)
        rec = tp / max(tp + fn, eps)
        f1 = 2 * prec * rec / max(prec + rec, eps)
        youden = rec - fp / max(fp + tn, eps)

        if prec >= min_precision and f1 > best_f1["f1"]:
            best_f1 = {"thr": float(t), "f1": f1}
        if youden > best_yj["youden"]:
            best_yj = {"thr": float(t), "youden": youden}

    return best_f1["thr"] if best_f1["f1"] >= 0 else best_yj["thr"]


# Group-aware dataset splitting
def build_group_splits(
    graphs: list[Data],
    y_values: np.ndarray,
    pocket_data: list[dict] | None = None,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    5-fold GroupKFold split with one inner fold for validation.

    Group assignment: uses ``protein_id`` or ``target_id`` graph attribute
    if present; otherwise derives a group from conf_id // 10 (coarse
    conformer grouping) to prevent leakage across states of the same protein.

    Returns
    -------
    train_idx, val_idx, test_idx, groups : np.ndarray
    """
    indices = np.arange(len(graphs))
    groups = np.array(
        [
            int(
                getattr(g, "protein_id", None) or getattr(g, "target_id", None) or (pocket_data[i]["conf_id"] // 10 if pocket_data else 0)
            )
            for i, g in enumerate(graphs)
        ]
    )

    outer = list(GroupKFold(n_splits=n_splits).split(indices, y_values, groups))
    trainval_idx, test_idx = outer[0]

    inner = list(
        GroupKFold(n_splits=n_splits).split(
            trainval_idx, y_values[trainval_idx], groups[trainval_idx]
        )
    )
    tr, va = inner[0]
    return trainval_idx[tr], trainval_idx[va], test_idx, groups


# Trainer
class Trainer:
    """
    Training and evaluation wrapper for AttentivePocketGNN.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        loss_fn: nn.Module | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn or FocalLoss(alpha=0.85, gamma=2.0)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5
        )
        self.best_metric = -float("inf")
        self.best_state: dict | None = None
        self.val_best_threshold: float = 0.5
        self.history: dict[str, list] = {
            "train_loss": [],
            "val_auroc": [],
            "val_auprc": [],
        }

    def _epoch(
        self, loader: DataLoader, train: bool = True
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        self.model.train(train)
        total_loss, total = 0.0, 0
        all_probs, all_tgts = [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                y = batch.y.view(-1).float()
                loss = self.loss_fn(logits, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item() * batch.num_graphs
                total += batch.num_graphs
                all_probs.append(torch.sigmoid(logits).detach().cpu())
                all_tgts.append(y.detach().cpu())

        probs = torch.cat(all_probs).numpy()
        tgts = torch.cat(all_tgts).numpy()

        try:
            auroc = float(roc_auc_score(tgts, probs))
        except ValueError:
            auroc = float("nan")
        try:
            auprc = float(average_precision_score(tgts, probs))
        except ValueError:
            auprc = float("nan")

        return total_loss / max(total, 1), auroc, auprc, probs, tgts

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 80,
        early_stopping_patience: int = 12,
        verbose_every: int = 5,
    ) -> dict[str, list]:
        no_improve = 0
        for ep in range(1, epochs + 1):
            tr_loss, _, _, _, _ = self._epoch(train_loader, train=True)
            _, val_auroc, val_auprc, vp, vt = self._epoch(val_loader, train=False)

            self.val_best_threshold = choose_threshold(vt, vp, min_precision=0.50)
            self.history["train_loss"].append(tr_loss)
            self.history["val_auroc"].append(val_auroc)
            self.history["val_auprc"].append(val_auprc)

            metric = val_auprc if not np.isnan(val_auprc) else -np.inf
            if metric > self.best_metric + 1e-5:
                self.best_metric = metric
                self.best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            self.scheduler.step(metric)

            if ep % verbose_every == 0 or ep == 1:
                print(
                    f"Epoch {ep:03d} | loss={tr_loss:.4f} | "
                    f"val_AUROC={val_auroc:.3f} | val_AUPRC={val_auprc:.3f} | "
                    f"thr={self.val_best_threshold:.3f}"
                )

            if no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {ep}")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.history

    # EVALS
    def evaluate(self, loader: DataLoader, threshold: float | None = None) -> dict:
        _, auroc, auprc, probs, tgts = self._epoch(loader, train=False)
        thr = threshold or self.val_best_threshold
        preds = (probs >= thr).astype(int)
        acc = float((preds == tgts).mean())
        try:
            tn, fp, fn, tp = confusion_matrix(tgts, preds, labels=[0, 1]).ravel()
        except ValueError:
            tn = fp = fn = tp = 0
        f1 = float(f1_score(tgts, preds, zero_division=0))
        return {
            "AUROC": auroc,
            "AUPRC": auprc,
            "ACC": acc,
            "F1": f1,
            "threshold": thr,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "probs": probs,
            "targets": tgts,
        }

    def evaluate_logits(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Return raw logits and targets (for Platt calibration)."""
        self.model.eval()
        logits_list, tgts_list = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits_list.append(self.model(batch).detach().cpu())
                tgts_list.append(batch.y.view(-1).float().detach().cpu())
        return torch.cat(logits_list).numpy(), torch.cat(tgts_list).numpy()

    # Platt calibration
    def calibrate(
        self, val_loader: DataLoader
    ) -> tuple[LogisticRegression, float, tuple[np.ndarray, np.ndarray]]:
        """
        Fit a Platt scaler (logistic regression on raw logits) to the
        validation set and return calibrated probabilities + best threshold.

        Returns
        -------
        platt : LogisticRegression
        best_threshold : float
        (val_probs_cal, val_targets) : tuple of arrays
        """
        val_logits, val_tgts = self.evaluate_logits(val_loader)
        platt = LogisticRegression(max_iter=1000, class_weight="balanced")
        platt.fit(val_logits.reshape(-1, 1), val_tgts)
        val_probs_cal = platt.predict_proba(val_logits.reshape(-1, 1))[:, 1]
        best_thr = choose_threshold(val_tgts, val_probs_cal, min_precision=0.50)
        return platt, float(best_thr), (val_probs_cal, val_tgts)
