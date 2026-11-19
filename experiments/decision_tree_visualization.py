#!/usr/bin/env python3
"""Render an illustrative decision tree for the accident-severity target.

This is a standalone visualization.  It fits a shallow DecisionTreeClassifier
on the same training rows used by the production pipeline, but it is **not**
the production Random Forest and it does not write to ``model/``.

Usage:
    python experiments/decision_tree_visualization.py
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from schema import TARGET  # noqa: E402
from train import build_pipeline, split_dataset  # noqa: E402

DATA_PATH = REPO_ROOT / "data" / "accidents.csv"
PNG_PATH = REPO_ROOT / "docs" / "eda" / "decision_tree.png"
RULES_PATH = REPO_ROOT / "experiments" / "decision_tree_rules.txt"

CLASS_NAMES = ["Non-severe", "Severe"]
CLASS_COLORS = ["#dbeafe", "#fecaca"]


def fit_tree(max_depth: int = 3, random_state: int = 42):
    """Fit the visualization tree on the training rows only."""
    frame = pd.read_csv(DATA_PATH)
    splits = split_dataset(
        frame,
        test_size=0.2,
        validation_size=0.2,
        random_state=random_state,
    )

    base = build_pipeline(
        {
            "algorithm": "rf",
            "n_estimators": 200,
            "max_depth": 10,
            "balancing": "none",
            "random_state": random_state,
        }
    )
    preprocessor = base.named_steps["preprocessor"]
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "tree",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(splits["x_train"], splits["y_train"])
    feature_names = [
        str(name).split("__", 1)[-1]
        for name in preprocessor.get_feature_names_out()
    ]
    return pipeline, feature_names, splits


def rules_text(pipeline: Pipeline, feature_names: list[str]) -> str:
    tree = pipeline.named_steps["tree"]
    body = export_text(tree, feature_names=feature_names, decimals=2)
    return "Class mapping: 0 = Non-severe, 1 = Severe\n\n" + body


def _place_nodes(tree) -> dict[int, tuple[float, int]]:
    positions: dict[int, tuple[float, int]] = {}
    next_leaf = [0]

    def place(node: int, depth: int) -> float:
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        if left == -1 and right == -1:
            x = float(next_leaf[0])
            next_leaf[0] += 1
        else:
            left_x = place(left, depth + 1)
            right_x = place(right, depth + 1)
            x = (left_x + right_x) / 2.0
        positions[node] = (x, -depth)
        return x

    place(0, 0)
    return positions


def save_tree_png(
    pipeline: Pipeline,
    feature_names: list[str],
    path: Path = PNG_PATH,
    max_depth: int = 3,
    train_rows: int | None = None,
) -> None:
    tree = pipeline.named_steps["tree"].tree_
    positions = _place_nodes(tree)
    n_leaves = int(tree.n_leaves)
    spacing = 3.2
    box_width = 3.0
    box_height = 1.08

    xs = [x * spacing for x, _ in positions.values()]
    x_min = min(xs) - box_width
    x_max = max(xs) + box_width
    fig_width = max(20.0, (x_max - x_min) * 0.95)
    fig, ax = plt.subplots(figsize=(fig_width, 12))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-(max_depth + 1.2), 1.1)
    ax.axis("off")

    for node, (x, y) in positions.items():
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        if left == -1 and right == -1:
            continue
        for child, label in ((left, "yes"), (right, "no")):
            child_x, child_y = positions[child]
            start = (x * spacing, y - box_height / 2)
            end = (child_x * spacing, child_y + box_height / 2)
            ax.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.3,
                    color="#64748b",
                    shrinkA=0,
                    shrinkB=0,
                )
            )
            ax.text(
                ((x + child_x) / 2) * spacing + (-0.18 if label == "yes" else 0.18),
                (y + child_y) / 2,
                label,
                ha="center",
                va="center",
                fontsize=10,
                color="#334155",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none"},
            )

    for node, (x, y) in positions.items():
        # scikit-learn >= 1.4 stores class proportions in ``tree_.value``.
        proportions = np.asarray(tree.value[node][0], dtype=float)
        samples = float(tree.weighted_n_node_samples[node])
        counts = np.round(proportions * samples).astype(int)
        if counts.sum() != int(round(samples)):
            counts[np.argmax(proportions)] += int(round(samples)) - int(counts.sum())
        majority = int(np.argmax(proportions))
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        is_leaf = left == -1 and right == -1

        if is_leaf:
            lines = [f"Predict: {CLASS_NAMES[majority]}"]
        else:
            feature = feature_names[int(tree.feature[node])]
            condition = f"{feature} <= {tree.threshold[node]:.2f}"
            lines = [textwrap.fill(condition, width=31)]

        lines.extend(
            [
                f"gini = {tree.impurity[node]:.3f}",
                f"samples = {int(tree.n_node_samples[node])}",
                f"Non-severe = {int(counts[0])}, Severe = {int(counts[1])}",
            ]
        )
        if not is_leaf:
            lines.append(f"class = {CLASS_NAMES[majority]}")

        box = FancyBboxPatch(
            (x * spacing - box_width / 2, y - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4,
            edgecolor="#334155",
            facecolor=CLASS_COLORS[majority],
        )
        ax.add_patch(box)
        ax.text(
            x * spacing,
            y,
            "\n".join(lines),
            ha="center",
            va="center",
            fontsize=8.6,
            linespacing=1.28,
            color="#0f172a",
        )

    row_text = f"trained on {train_rows:,} training rows" if train_rows else "trained on training rows only"
    fig.suptitle(
        f"Illustrative decision tree (max_depth={max_depth}) — {row_text}",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Standalone illustration, not the production Random Forest. "
        "Left branch = yes, right branch = no.",
        ha="center",
        fontsize=10,
        color="#475569",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an illustrative decision tree.")
    parser.add_argument("--max-depth", type=int, default=3, choices=[3, 4])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--png", type=Path, default=PNG_PATH)
    parser.add_argument("--rules", type=Path, default=RULES_PATH)
    args = parser.parse_args()

    pipeline, feature_names, splits = fit_tree(args.max_depth, args.random_state)
    save_tree_png(
        pipeline,
        feature_names,
        args.png,
        args.max_depth,
        len(splits["y_train"]),
    )
    rules = rules_text(pipeline, feature_names)
    args.rules.parent.mkdir(parents=True, exist_ok=True)
    args.rules.write_text(rules)

    print(rules)
    print(f"Tree image: {args.png}")
    print(f"Rules text: {args.rules}")


if __name__ == "__main__":
    main()
