"""Detection model builders for Exercise 3."""

from Exercise3.models.faster_rcnn import (
    FasterRCNNBaselineConfig,
    build_faster_rcnn_baseline,
    configure_model_for_training,
    summarize_faster_rcnn,
)

__all__ = [
    "FasterRCNNBaselineConfig",
    "build_faster_rcnn_baseline",
    "configure_model_for_training",
    "summarize_faster_rcnn",
]
