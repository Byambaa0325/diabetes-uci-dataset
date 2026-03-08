"""Public API for src.pipeline.

    from src.pipeline import run_pipeline, run_pipeline_from_features, sweep, load_model
    from src.pipeline import stage_load, stage_featurize, stage_train, stage_evaluate
    from src.pipeline import DataBundle, FeatureBundle, PipelineResult
"""
from src.pipeline.types import DataBundle, FeatureBundle, PipelineResult
from src.pipeline.load import stage_load
from src.pipeline.featurize import stage_featurize
from src.pipeline.run_train import stage_train
from src.pipeline.run_evaluate import stage_evaluate
from src.pipeline.run_test import stage_test
from src.pipeline.runner import run_pipeline, run_pipeline_from_features, sweep, load_model
from src.pipeline.cross_validate import cross_validate, cross_validate_sweep

__all__ = [
    "DataBundle", "FeatureBundle", "PipelineResult",
    "stage_load", "stage_featurize", "stage_train", "stage_evaluate", "stage_test",
    "run_pipeline", "run_pipeline_from_features", "sweep", "load_model",
    "cross_validate", "cross_validate_sweep",
]
