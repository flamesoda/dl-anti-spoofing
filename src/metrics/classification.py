import torch

from src.metrics.base_metric import BaseMetric


class BinaryAccuracy(BaseMetric):
    """Batch accuracy for the two-class CM problem."""

    def __call__(self, logits, labels, **batch):
        return (logits.argmax(dim=-1) == labels).float().mean().item()
