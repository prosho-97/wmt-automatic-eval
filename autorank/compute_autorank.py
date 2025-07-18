import csv
import json
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to compute AutoRank on a language pair specified in input."
    )

    parser.add_argument(
        "--metrics-outputs-path",
        type=Path,
        required=True,
        help="[REQUIRED] Path to the directory containing the MT metrics outputs. It must contain one directory for "
        "each MT metric.",
    )

    parser.add_argument(
        "--teams-path",
        type=Path,
        required=True,
        help="[REQUIRED] Path to the json file containing the scored MT systems info.",
    )

    parser.add_argument(
        "--lp",
        type=str,
        default="en-cs_CZ",
        help="Which language pair to run AutoRank on. It must be passed in local code format (e.g., 'en-et_EE'). "
        "Default: 'en-cs_CZ'.",
    )

    parser.add_argument(
        "--domain",
        type=str,
        help="If passed, the AutoRank will be computed considering only the specified domain. ",
    )

    parser.add_argument(
        "--out-path",
        type=Path,
        required=True,
        help="[REQUIRED] Path to the directory where the AutoRank results will be stored.",
    )

    return parser


def robust_scale_metric(
    sys2metric_score: Dict[str, float],
    lower_pct: float = 25.0,
    upper_pct: float = 100.0,
    higher_is_better: bool = True,
) -> Dict[str, float]:
    """
    Apply Robust Scaling to the input metric scores.

    Args:
        sys2metric_score: Dictionary mapping system names to their metric scores.
        lower_pct: Lower percentile to use for scaling. Default: 25.0.
        upper_pct: Upper percentile to use for scaling. Default: 100.0.
        higher_is_better: If True, higher scores are better. Default: True.

    Returns:
        A dictionary mapping each system name to its robust scaled metric score.
    """
    systems = list(sys2metric_score)
    scores = np.array([sys2metric_score[sys] for sys in systems], dtype=float)

    q_low = np.percentile(scores, lower_pct)
    q_high = np.percentile(scores, upper_pct)
    scale = q_high - q_low or 1.0  # To avoid division by zero.

    z_scores = (scores - np.median(scores)) / scale

    if not higher_is_better:
        z_scores *= -1

    return {sys: float(z) for sys, z in zip(systems, z_scores)}


def average_and_rank(
    sys2robust_scaled_metric_scores: Dict[str, List[float]]
) -> Dict[str, float]:
    """
    Compute AutoRank scores based on the average of robust scaled metric scores.

    Args:
        sys2robust_scaled_metric_scores: Dictionary mapping system names to lists of robust scaled metric scores.

    Returns:
        A dictionary mapping each system name to its AutoRank score.
    """
    systems = list(sys2robust_scaled_metric_scores)
    z_matrix = np.array(
        [sys2robust_scaled_metric_scores[sys] for sys in systems], dtype=float
    )

    avg_z = z_matrix.mean(axis=1)

    z_min, z_max = avg_z.min(), avg_z.max()
    N = len(systems)
    autorank = 1 + (N - 1) * (z_max - avg_z) / (z_max - z_min)

    return dict(zip(systems, autorank))


def compute_autorank() -> None:
    """
    Command to compute AutoRank on a language pair specified in input.
    """
    args = read_arguments().parse_args()

    metric_name2outputs = dict()
    for metric_dir in args.metrics_outputs_path.iterdir():
        if metric_dir.is_dir():
            metric_outputs_file = metric_dir / "outputs.pickle"
            if metric_outputs_file.exists():
                with open(metric_outputs_file, "rb") as f:
                    metric_name2outputs[metric_dir.name] = pickle.load(f)

    with open(args.teams_path, "r", encoding="utf-8") as f:
        teams = json.load(f)
    sys2is_constrained = dict()
    for entry in teams:
        if entry["publication_name"] in sys2is_constrained:
            raise ValueError(
                f"Publication name {entry['publication_name']} is not unique in the teams file."
            )
        assert len(entry["primary_submissions"]) == 1
        sys2is_constrained[entry["publication_name"]] = entry["primary_submissions"][0][
            "is_constrained"
        ]

    sys2robust_scaled_metric_scores = defaultdict(list)
    for metric, sys2lp_scores in metric_name2outputs.items():
        sys2scores = dict()
        for sys, lp2domain_scores in sys2lp_scores.items():
            for domain, doc_id2scores in lp2domain_scores[args.lp].items():
                if args.domain is not None and domain != args.domain:
                    continue
                for scores in doc_id2scores.values():
                    if sys not in sys2scores:
                        sys2scores[sys] = []
                    sys2scores[sys] += scores
            sys2scores[sys] = sum(sys2scores[sys]) / len(sys2scores[sys])

        with open(args.output_path / f"{metric}.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(["Sys", "Is Constrained?", f"{metric} Score"])

            for sys, score in sorted(
                sys2scores.items(), key=lambda item: item[1], reverse=True
            ):
                writer.writerow(
                    (sys, "Yes" if sys2is_constrained[sys] else "No", round(score, 4))
                )

        for sys, robust_scaled_score in robust_scale_metric(sys2scores).items():
            sys2robust_scaled_metric_scores[sys].append(robust_scaled_score)

    # Compute the final AutoRank as the average of robust scaled scores across all metrics.
    with open(args.output_path / "autorank.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["Sys", "Is Constrained?", "AutoRank"])

        for sys, autorank_score in sorted(
            average_and_rank(sys2robust_scaled_metric_scores).items(),
            key=lambda item: item[1],
            reverse=False,
        ):
            writer.writerow(
                (
                    sys,
                    "Yes" if sys2is_constrained[sys] else "No",
                    round(autorank_score, 2),
                )
            )


if __name__ == "__main__":
    compute_autorank()
