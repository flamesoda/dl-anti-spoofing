#!/usr/bin/env python3
"""Create report-ready loss and EER plots from a WandB CSV export."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_non_null(axis, frame, x_column, y_column, label):
    subset = frame[[x_column, y_column]].dropna()
    if subset.empty:
        raise ValueError(f"Column '{y_column}' has no values")
    axis.plot(subset[x_column], subset[y_column], label=label, linewidth=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path, help="CSV exported from WandB")
    parser.add_argument("--output", type=Path, default=Path("training_history.png"))
    parser.add_argument("--step-column", default="_step")
    parser.add_argument("--train-loss", default="loss_train")
    parser.add_argument("--dev-loss", default="loss_dev")
    parser.add_argument("--dev-eer", default="EER_dev")
    args = parser.parse_args()

    history = pd.read_csv(args.history)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_non_null(axes[0], history, args.step_column, args.train_loss, "Train")
    plot_non_null(axes[0], history, args.step_column, args.dev_loss, "Development")
    axes[0].set(title="Cross-entropy loss", xlabel="Training step", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    plot_non_null(axes[1], history, args.step_column, args.dev_eer, "Development")
    axes[1].set(title="Development EER", xlabel="Training step", ylabel="EER, %")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
