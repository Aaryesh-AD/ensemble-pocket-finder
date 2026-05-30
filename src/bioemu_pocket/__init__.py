"""
bioemu-pocket-discovery
=======================
Ensemble-based druggable pocket detection using BioEMU conformational
ensembles and Attentive Graph Neural Networks.
"""

from bioemu_pocket.pocket_finder import EnsemblePocketFinder
from bioemu_pocket.graph_builder import PocketGraphBuilder
from bioemu_pocket.model import AttentivePocketGNN
from bioemu_pocket.trainer import Trainer

__all__ = [
    "EnsemblePocketFinder",
    "PocketGraphBuilder",
    "AttentivePocketGNN",
    "Trainer",
]

__version__ = "0.1.0"
