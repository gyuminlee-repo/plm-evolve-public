"""In-silico simulation harness for the EVOLVEpro active-learning loop.

This subpackage is the measurement foundation: it lets us run the SAME
train -> predict -> rank loop as the real wet-lab core against an oracle
landscape instead of an experiment, and quantify SAMPLE EFFICIENCY (how few
measurements to reach a target) with an explicit single-round baseline so that
"iterative beats single-round" is directly measurable.

Pieces:
    * oracle        -- Landscape (oracle map) + loaders + synthetic_landscape.
    * strategies    -- RandomStrategy, GreedyTopLayer (EVOLVEpro-equivalent,
                       reuses the core regressor), AdditiveLinear (null baseline).
    * simulate      -- run_simulation, single_round_simulation, run_panel.
    * metrics       -- sample-efficiency-first metrics.
    * embeddings_sim -- honest locality-swept synthetic embedding tables (GAP-2).
    * multiobjective -- scalarize/pareto_front/compose_objectives: opt-in
                       multi-objective HOOK (GAP-5) that composes 2+ objectives
                       into ONE scalar Landscape so the loop runs unchanged.
    * ablation      -- run_locality_ablation: isolate POOLING in the AL loop.
"""

from __future__ import annotations

from .oracle import Landscape, synthetic_landscape
from .strategies import (
    AdditiveLinear,
    ConstantLiarStrategy,
    CoverageBatchStrategy,
    GreedyTopLayer,
    RandomStrategy,
    ThompsonStrategy,
    UCBStrategy,
    STRATEGY_REGISTRY,
    build_strategy,
)
from .simulate import run_simulation, single_round_simulation, run_panel
from .metrics import (
    compute_round_metrics,
    fraction_top_bin,
    max_activity_found,
    rounds_to_target,
    simple_regret,
    spearman,
    top_k_recall,
)
from .embeddings_sim import (
    POOLINGS,
    pooling_gain,
    synthetic_embedding_tables,
)
from .multimutant import (
    AdditiveMultiPredictor,
    canonical_multi_token,
    enumerate_multimutants,
    make_generative_landscape,
    propose_multimutants,
)
from .multiobjective import (
    ScalarizedLandscape,
    compose_objectives,
    pareto_front,
    scalarize,
)
from .ablation import (
    build_decoy_cluster_landscape,
    build_redundant_landscape,
    build_rugged_landscape,
    run_acquisition_ablation,
    run_batch_diversity_ablation,
    run_locality_ablation,
    run_multimutant_ablation,
    run_multiobjective_ablation,
)

__all__ = [
    "Landscape",
    "synthetic_landscape",
    "RandomStrategy",
    "GreedyTopLayer",
    "AdditiveLinear",
    "UCBStrategy",
    "ThompsonStrategy",
    "CoverageBatchStrategy",
    "ConstantLiarStrategy",
    "STRATEGY_REGISTRY",
    "build_strategy",
    "run_simulation",
    "single_round_simulation",
    "run_panel",
    "compute_round_metrics",
    "max_activity_found",
    "simple_regret",
    "rounds_to_target",
    "top_k_recall",
    "fraction_top_bin",
    "spearman",
    "POOLINGS",
    "pooling_gain",
    "synthetic_embedding_tables",
    "run_locality_ablation",
    "build_rugged_landscape",
    "run_acquisition_ablation",
    "build_redundant_landscape",
    "build_decoy_cluster_landscape",
    "run_batch_diversity_ablation",
    "AdditiveMultiPredictor",
    "canonical_multi_token",
    "enumerate_multimutants",
    "make_generative_landscape",
    "propose_multimutants",
    "run_multimutant_ablation",
    "scalarize",
    "pareto_front",
    "compose_objectives",
    "ScalarizedLandscape",
    "run_multiobjective_ablation",
]
