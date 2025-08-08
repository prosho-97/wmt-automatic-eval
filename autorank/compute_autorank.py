import os
import csv
import json
import re

import ipdb
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


LANG_CODE_2_LANG_COUNTRY = {
    "cs": {
        "language": "Czech",
    },
    "ja": {
        "language": "Japanese",
    },
    "en": {
        "language": "English",
    },
    "cs_CZ": {
        "language": "Czech",
        "country": "Czechia",
    },
    "ar_EG": {
        "language": "Egyptian Arabic",
        "country": "Egypt",
    },
    "bn_BD": {
        "language": "Bengali",
        "country": "Bangladesh",
    },
    "bho_IN": {
        "language": "Bhojpuri",
        "country": "India",
    },
    "zh_CN": {
        "language": "Simplified Chinese",
        "country": "China",
    },
    "et_EE": {
        "language": "Estonian",
        "country": "Estonia",
    },
    "fa_IR": {
        "language": "Persian",
        "country": "Iran",
    },
    "de_DE": {
        "language": "German",
        "country": "Germany",
    },
    "el_GR": {
        "language": "Greek",
        "country": "Greece",
    },
    "hi_IN": {
        "language": "Hindi",
        "country": "India",
    },
    "is_IS": {
        "language": "Icelandic",
        "country": "Iceland",
    },
    "id_ID": {
        "language": "Indonesian",
        "country": "Indonesia",
    },
    "it_IT": {
        "language": "Italian",
        "country": "Italy",
    },
    "ja_JP": {
        "language": "Japanese",
        "country": "Japan",
    },
    "kn_IN": {
        "language": "Kannada",
        "country": "India",
    },
    "ko_KR": {
        "language": "Korean",
        "country": "South Korea",
    },
    "lt_LT": {
        "language": "Lithuanian",
        "country": "Lithuania",
    },
    "mr_IN": {
        "language": "Marathi",
        "country": "India",
    },
    "mas_KE": {
        "language": "Maasai",
        "country": "Kenya",
    },
    "ro_RO": {
        "language": "Romanian",
        "country": "Romania",
    },
    "ru_RU": {
        "language": "Russian",
        "country": "Russia",
    },
    "sr_Latn_RS": {
        "language": "Serbian (Latin)",
        "country": "Serbia",
    },
    "sr_Cyrl_RS": {
        "language": "Serbian (Cyrilics)",
        "country": "Serbia",
    },
    "sv_SE": {
        "language": "Swedish",
        "country": "Sweden",
    },
    "th_TH": {
        "language": "Thai",
        "country": "Thailand",
    },
    "tr_TR": {
        "language": "Turkish",
        "country": "Turkey",
    },
    "uk_UA": {
        "language": "Ukrainian",
        "country": "Ukraine",
    },
    "vi_VN": {
        "language": "Vietnamese",
        "country": "Vietnam",
    },
}


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
        default=Path("../data/wmt25-general-mt/data/systems_metadata_updated3.json"),
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


def will_be_human_evaluated(df: pd.DataFrame, lp) -> pd.Series:
    df["will_humeval"] = False
    contrained_limit = 8
    total = 18
    if lp == "en-cs_CZ":
        contrained_limit = 9
        total = 19
    elif lp == "cs-de_DE":
        contrained_limit = 10
        total = 20
    elif lp == "en-ko_KR":
        total = 19

    constrained = df[df["is_constrained"] == True].head(contrained_limit)
    df.loc[constrained.index, "will_humeval"] = True
    for idx, row in df.iterrows():
        forbidden = ["bb88", "ctpc_nlp", "MMMT"]
        if idx in forbidden:
            print(f"Skipping {idx} as it is in the forbidden list.")
            continue
        df.loc[idx, "will_humeval"] = True
        if len(df[df["will_humeval"]]) >= total:
            break

    return df


def normalize_param_count(val: Optional[str] = None) -> str:
    """
    Normalize parameter count values for display (billions).
    - Accepts strings like '14B', '633.2M', '4 * 14B', etc.
    - Converts millions to billions.
    - Handles numbers, dashes, Nones, and vague values.

    Args:
        val: The string value to normalize.

    Returns:
        A string representing the normalized value in billions, or "not specified" if the input is invalid.
    """
    if val is None or str(val).strip() in {"", "-", "several billion"}:
        return "Unknown"

    s = str(val).strip()

    # Try parsing arithmetic expressions like "4 * 14B"
    if "*" in s:
        try:
            # Replace B/b/M/m with *1e9 or *1e6 for safe eval
            s_eval = re.sub(r"(\d+(?:\.\d+)?)[Bb]", r"(\1*1e9)", s)
            s_eval = re.sub(r"(\d+(?:\.\d+)?)[Mm]", r"(\1*1e6)", s_eval)
            num = eval(s_eval)
            return f"{num / 1e9:.0f}".rstrip(".0")
        except Exception:
            return "Unknown"

    # Handle '<1'
    if s == "<1":
        return "<1"

    # Handle values ending with B/b (billions)
    match_b = re.match(r"^(\d+(?:\.\d+)?)\s*[Bb]$", s)
    if match_b:
        return str(float(match_b.group(1))).rstrip(".0")

    # Handle values ending with M/m (millions)
    match_m = re.match(r"^(\d+(?:\.\d+)?)\s*[Mm]$", s)
    if match_m:
        val_b = float(match_m.group(1)) / 1000
        return f"{val_b:.2f}".rstrip("0").rstrip(".")

    # Pure numbers: treat as billions if reasonable
    try:
        num = float(s)
        if num > 1000:  # Unlikely to be billions, probably millions
            return f"{num / 1000:.2f}".rstrip("0").rstrip(".")
        return str(num).rstrip(".0")
    except Exception:
        pass

    # If nothing matches, return 'not specified'
    return "Unknown"


def escape_latex(text: str) -> str:
    """
    Escape LaTeX special characters in a string.

    Args:
        text: The input string to escape. If not a string, it is returned unchanged.

    Returns:
        A string with LaTeX special characters escaped, or the original input if it is not a string.
    """
    if not isinstance(text, str):
        return text
    conv = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    regex = re.compile("|".join(re.escape(key) for key in conv.keys()))
    return regex.sub(lambda match: conv[match.group()], text)


def generate_latex_table(
    df: pd.DataFrame,
    teams: dict,
    language_pair: str,
    output_path,
) -> None:
    """
    Generate a LaTeX table for a language pair with system metadata.

    Args:
        df: DataFrame containing system scores with columns for metrics and 'autorank'.
        teams: Dictionary mapping system names to metadata (e.g., is constrained, parameter count).
        language_pair: Language pair in the format 'en-cs_CZ'.
        output_path: Path to save the generated LaTeX table.
    """

    def get_pretty_lang_pair_name(language_pair: str) -> str:
        """
        Get a human-readable name for the language pair.

        Args:
            language_pair: Language pair in the format 'en-cs_CZ'.

        Returns:
            A string representing the language pair in a human-readable format.
        """
        assert "-" in language_pair
        src, tgt = language_pair.split("-")

        # Try to map with LANG_CODE_2_LANG_COUNTRY
        def lookup_name(code: str) -> str:
            """
            Look up the language name for a given code.

            Args:
                code: Language code (e.g., 'en', 'cs_CZ').

            Returns:
                The language name if found, otherwise returns the code itself.
            """
            entry = LANG_CODE_2_LANG_COUNTRY.get(code)
            if entry and "language" in entry:
                return entry["language"]
            # Try to remove the region (e.g., en-cs_CZ → cs_CZ)
            if "_" in code:
                entry = LANG_CODE_2_LANG_COUNTRY.get(code.split("_")[0])
                if entry and "language" in entry:
                    return entry["language"]
            return code

        src_name = lookup_name(src)
        tgt_name = lookup_name(tgt)
        return f"{src_name}-{tgt_name}"

    # Only consider systems present in both df and metadata
    systems_to_show = [sys for sys in df.index if sys in teams]

    table_rows = []
    rowcolors = []  # Track if the row should be gray (unconstrained)

    # All raw metric columns (in order of appearance)
    metric_cols = [c for c in df.columns if c.endswith("_raw")]

    table_data = []
    for sys in systems_to_show:
        meta = teams[sys]

        # System name
        team_name_escaped = escape_latex(sys)

        # LP supported
        supported_lps = meta.get("supported_lps", {})
        lp_support = supported_lps.get(language_pair)
        if lp_support == "supported":
            lp_mark = r"\checkmark"
        elif lp_support == "unsupported":
            lp_mark = r"\ding{55}"  # or r"\texttimes" or "--"
        else:
            lp_mark = "Unknown"

        # Parameter count (normalized, in billions)
        param_count = normalize_param_count(meta.get("parameter_count"))

        # AutoRank score
        autorank = df.loc[sys]["autorank"]
        autorank_str = f"{round(autorank, 1)}"

        # Raw metric scores
        metric_vals = []
        for col in metric_cols:
            val = df.loc[sys][col]
            metric_vals.append(f"{round(val, 3 if 'comet' in col.lower() else 1)}")

        # Human evaluation column: ✓ if True, else blank
        humeval = df.loc[sys]["will_humeval"]
        humeval_str = r"\checkmark" if bool(humeval) else ""

        row = (
            [team_name_escaped, lp_mark, param_count, autorank_str]
            + metric_vals
            + [humeval_str]
        )
        table_rows.append(row)

        # Save as tuple: (raw_name, row_data, is_unconstrained)
        is_unconstrained = not meta.get("constrained", False)
        table_data.append((sys, row, is_unconstrained))

    # Build header (with arrows)
    header = (
        ["System Name", "LP Supported", "Param. Count (B)", "AutoRank $\\downarrow$"]
        + [f"{c.replace('_raw', '')} $\\uparrow$" for c in metric_cols]
        + ["Human Evaluation?"]
    )

    # Tabular column spec: one col for each field
    ncols = 4 + len(metric_cols) + 1
    colspec = "l" + "X" * (ncols - 1)

    latex_df = pd.DataFrame(
        [d[1] for d in table_data], columns=header, index=[d[0] for d in table_data]
    )
    # Sort by AutoRank (ascending)
    try:
        latex_df["AutoRank $\\downarrow$"] = latex_df["AutoRank $\\downarrow$"].astype(
            float
        )
        latex_df = latex_df.sort_values("AutoRank $\\downarrow$", ascending=True)
    except Exception:
        latex_df = latex_df

    # Compose LaTeX lines
    latex_lines = []
    latex_lines.append(r"\usepackage[table]{xcolor}")
    latex_lines.append(r"\usepackage{booktabs}")
    latex_lines.append("")
    # Use tabularx for auto-resizing
    pretty_title = get_pretty_lang_pair_name(language_pair)
    latex_lines.append(r"\rowcolors{2}{gray!20}{white}")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabularx}{\textwidth}{" + colspec + "}")
    latex_lines.append(r"\toprule")
    latex_lines.append(rf"\multicolumn{{{ncols}}}{{c}}{{\textbf{{{pretty_title}}}}} \\")
    latex_lines.append(r"\midrule")
    latex_lines.append(" & ".join(header) + r" \\")
    latex_lines.append(r"\midrule")

    # Build a mapping from sys (raw name) to is_unconstrained
    unconstr_map = dict((d[0], d[2]) for d in table_data)
    for idx, row in latex_df.iterrows():
        row_data = row.tolist()
        color_cmd = ""
        if unconstr_map.get(idx, False):
            color_cmd = r"\rowcolor{gray!30}"
        row_str = " & ".join(str(x) for x in row_data) + r" \\"
        if color_cmd:
            latex_lines.append(color_cmd)
        latex_lines.append(row_str)

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabularx}")

    # Write to the output file
    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(latex_lines))

    print(f"LaTeX table written to {output_path}")


reference_exists = [
    "cs-uk_UA",
    "en-ar_EG",
    "en-cs_CZ",
    "en-et_EE",
    "en-is_IS",
    "en-ja_JP",
    "en-ko_KR",
    "en-ru_RU",
    "en-uk_UA",
    "en-zh_CN",
    "ja-zh_CN",
    "cs-de_DE",
    "en-sr_Cyrl_RS",
]

chrf_only = ["en-bho_IN", "en-mas_KE"]


def compute_autorank(language_pair, args) -> None:
    """
    Command to compute AutoRank on a language pair specified in input.
    """

    metric_name2outputs = dict()
    for metric_dir in args.metrics_outputs_path.iterdir():
        if "CometKiwi-XL" in metric_dir.name and language_pair not in reference_exists:
            # Skip CometKiwi-XL for language pairs without reference translations because xCOMET and MetricX are used in QE mode
            continue

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
            if sys not in teams or language_pair not in teams[sys]["supported_lps"]:
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
                if metric == "chrF++" or language_pair not in chrf_only:
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
    df = will_be_human_evaluated(df, language_pair)

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

    generate_latex_table(
        df,
        teams,
        language_pair,
        output_path=args.out_path / f"autorank_{language_pair}.tex",
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
