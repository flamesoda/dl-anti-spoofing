import numpy as np
import torch

from src.metrics.base_metric import BaseMetric


def compute_det_curve(target_scores, nontarget_scores):
    """Compute the official ASVspoof false-rejection/acceptance curves."""
    target_scores = np.asarray(target_scores, dtype=np.float64)
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64)
    if target_scores.ndim != 1 or nontarget_scores.ndim != 1:
        raise ValueError("EER scores must be one-dimensional")
    if target_scores.size == 0 or nontarget_scores.size == 0:
        raise ValueError("Both bona fide and spoof scores are required")
    combined = np.concatenate((target_scores, nontarget_scores))
    if not np.isfinite(combined).all():
        raise ValueError("EER scores contain NaN or infinity")

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )
    return frr, far, thresholds


def compute_eer(bonafide_scores, other_scores):
    """Return EER as a fraction and its threshold using the course code."""
    frr, far, thresholds = compute_det_curve(bonafide_scores, other_scores)
    min_index = np.argmin(np.abs(frr - far))
    eer = np.mean((frr[min_index], far[min_index]))
    return float(eer), float(thresholds[min_index])


class EqualErrorRate(BaseMetric):
    """Pooled, epoch-level EER on the 0--100 percentage scale."""

    is_epoch_metric = True

    def __init__(self, bonafide_label=1, name="EER"):
        super().__init__(name=name)
        self.bonafide_label = bonafide_label
        self.reset()

    def reset(self):
        self._scores = []
        self._labels = []

    def update(self, scores, labels, **batch):
        self._scores.append(scores.detach().cpu().to(torch.float64))
        self._labels.append(labels.detach().cpu().to(torch.long))

    def compute(self):
        if not self._scores:
            raise RuntimeError("EER was requested before any scores were accumulated")
        scores = torch.cat(self._scores).numpy()
        labels = torch.cat(self._labels).numpy()
        bona_scores = scores[labels == self.bonafide_label]
        spoof_scores = scores[labels != self.bonafide_label]
        eer, _ = compute_eer(bona_scores, spoof_scores)
        return eer * 100.0

    def __call__(self, **batch):
        raise RuntimeError(
            "EqualErrorRate is an epoch-level metric; Trainer must call update/compute"
        )
