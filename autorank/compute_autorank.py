import os
import csv
import json
import ipdb
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to compute AutoRank on a language pair specified in input."
    )

    parser.add_argument(
        "--metrics-outputs-path",
        type=Path,
        default=Path("../data/metrics_outputs"),
        help="Path to the directory containing the MT metrics outputs. It must contain one directory for "
        "each MT metric.",
    )

    parser.add_argument(
        "--teams-path",
        type=Path,
        default=Path("../data/wmt25-general-mt/data/systems_metadata.json"),
        help="Path to the json file containing the scored MT systems info.",
    )

    parser.add_argument(
        "--lps",
        type=str,
        nargs="+",
        default=["en-cs_CZ"],
        help="Which language pair(s) to run AutoRank on. Must be passed in local code format (e.g., 'en-et_EE'). "
        "Multiple language pairs can be provided as space-separated values. "
        "Default: en-cs_CZ",
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
    # Compute the average for each system (handle empty lists gracefully)
    avg_z = np.array(
        [
            np.mean(sys2robust_scaled_metric_scores[sys])
            if sys2robust_scaled_metric_scores[sys]
            else float("-inf")
            for sys in systems
        ],
        dtype=float,
    )

    # Remove systems with no scores (avg == -inf)
    valid_idx = avg_z != float("-inf")
    systems = [s for i, s in enumerate(systems) if valid_idx[i]]
    avg_z = avg_z[valid_idx]

    if len(avg_z) == 0:
        return dict()

    z_min, z_max = avg_z.min(), avg_z.max()
    N = len(systems)

    # If all values are equal, set all autorank to 1.0
    if np.isclose(z_max, z_min):
        autorank = np.ones(N)
    else:
        autorank = 1 + (N - 1) * (z_max - avg_z) / (z_max - z_min)

    return dict(zip(systems, autorank))


def will_be_human_evaluated(df: pd.DataFrame) -> pd.Series:
    df["will_humeval"] = False
    constrained = df[df["is_constrained"] == True].head(8)
    df.loc[constrained.index, "will_humeval"] = True
    for idx, row in df.iterrows():
        forbidden = ["bb88", "ctpc_nlp", "MMMT", "TranssionTranslate"]
        if idx in forbidden:
            print(f"Skipping {idx} as it is in the forbidden list.")
            continue
        df.loc[idx, "will_humeval"] = True
        if len(df[df["will_humeval"]]) >= 18:
            break

    return df


def compute_autorank(language_pair, args) -> None:
    """
    Command to compute AutoRank on a language pair specified in input.
    """

    metric_name2outputs = dict()
    for metric_dir in args.metrics_outputs_path.iterdir():
        if metric_dir.is_dir() and not (
            metric_dir.name == "chrF++"
            and language_pair != "en-bho_IN"
            and language_pair != "en-mas_KE"
        ):
            metric_outputs_file = metric_dir / "outputs.pickle"
            if metric_outputs_file.exists():
                with open(metric_outputs_file, "rb") as f:
                    metric_name2outputs[metric_dir.name] = pickle.load(f)

    with open(args.teams_path, "r", encoding="utf-8") as f:
        teams = json.load(f)

    sys2robust_scaled_metric_scores = defaultdict(list)
    system_scores = defaultdict(dict)
    for metric, sys2lp_scores in metric_name2outputs.items():
        sys2scores = dict()
        for sys, lp2domain_scores in sys2lp_scores.items():
            if sys not in teams or language_pair not in teams[sys]["lps"]:
                continue
            if (
                language_pair not in lp2domain_scores
            ):  # sys did not submit translations for this lp
                continue
            scores_sum, n_valid_scores, n_scores = 0, 0, 0
            for domain, doc_id2scores in lp2domain_scores[language_pair].items():
                if args.domain is not None and domain != args.domain:
                    continue
                for scores in doc_id2scores.values():
                    for s in scores:
                        if s is not None:
                            scores_sum += s
                            n_valid_scores += 1
                        n_scores += 1
            if n_valid_scores != n_scores:
                print(
                    f"Found {n_scores - n_valid_scores} None scores out of {n_scores} for {sys} in "
                    f"{metric}. Percentage: {round(((n_scores - n_valid_scores) / n_scores) * 100, 2)}%"
                )
            sys2scores[sys] = scores_sum / n_valid_scores

        if len(sys2scores) > 0:
            with open(args.out_path / f"{metric}.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                writer.writerow(["Sys", "Is Constrained?", f"{metric} Score"])

                for sys, score in sorted(
                    sys2scores.items(), key=lambda item: item[1], reverse=True
                ):
                    writer.writerow(
                        (
                            sys,
                            "Yes" if teams[sys]["constrained"] else "No",
                            round(score, 4),
                        )
                    )

            for sys, robust_scaled_score in robust_scale_metric(sys2scores).items():
                sys2robust_scaled_metric_scores[sys].append(robust_scaled_score)
                system_scores[sys][metric] = robust_scaled_score
                system_scores[sys][f"{metric}_raw"] = sys2scores[sys]
                system_scores[sys]["is_constrained"] = teams[sys]["constrained"]
        else:
            print(
                f"Warning: No scores found for metric {metric} on language pair {language_pair}. Skipping."
            )

    df = pd.DataFrame(system_scores).T
    # add autorank based on average_and_rank(sys2robust_scaled_metric_scores) dictionary
    df["autorank"] = average_and_rank(sys2robust_scaled_metric_scores)
    # sort by autorank
    df = df.sort_values(by="autorank", ascending=True)
    df = will_be_human_evaluated(df)

    # sort column in order: is_constrained, autorank, all scaled metrics, all raw metrics
    cols = ["is_constrained", "will_humeval", "autorank"]
    cols += sorted(
        [col for col in df.columns if not col.endswith("_raw") and col not in cols],
        key=lambda x: x.split("_")[0],
    )
    cols += sorted(
        [col for col in df.columns if col.endswith("_raw")],
        key=lambda x: x.split("_")[0],
    )
    df = df[cols]

    # Compute the final AutoRank as the average of robust scaled scores across all metrics.
    with open(args.out_path / "autorank.csv", "w", newline="") as csvfile:
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
                    "Yes" if teams[sys]["constrained"] else "No",
                    round(autorank_score, 2),
                )
            )

    return df


if __name__ == "__main__":
    args = read_arguments().parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    dfs = {}
    for lp in args.lps:
        lpdf = compute_autorank(lp, args)
        dfs[lp] = lpdf

    with pd.ExcelWriter(args.out_path / "autorank.xlsx") as writer:
        for lp, df in dfs.items():
            df.to_excel(writer, sheet_name=lp, index=True, float_format="%.2f")
