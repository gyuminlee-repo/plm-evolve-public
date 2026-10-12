"""plm_evolve: independent implementation of the EVOLVEpro core principle.

Frozen protein-language-model embeddings feed a top-layer RandomForest
regressor trained on a few measured variants. The model predicts every
unmeasured single mutant, ranks them, and proposes the next experimental
round (active learning). This package implements ONLY that core loop.
"""

from .embeddings import (
    EmbeddingExtractor,
    ESM2Extractor,
    ESMCExtractor,
    make_extractor,
)
from .variants import parse_variant, generate_single_mutant_fasta, AA_ALPHABET
from .rounds import load_round_data, build_training_table, validate_coverage
from .regressor import TopLayerRegressor
from .evolve import run_round, RoundResult

__version__ = "0.1.0"

__all__ = [
    "EmbeddingExtractor",
    "ESM2Extractor",
    "ESMCExtractor",
    "make_extractor",
    "parse_variant",
    "generate_single_mutant_fasta",
    "AA_ALPHABET",
    "load_round_data",
    "build_training_table",
    "validate_coverage",
    "TopLayerRegressor",
    "run_round",
    "RoundResult",
    "__version__",
]
