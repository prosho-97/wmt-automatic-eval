import json
import logging
import pickle
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from typing import Dict, List

from metrics.comet.comet_scorer import CometScorer
from metrics.gemba.gemba_scorer import GembaScorer
from metrics.metricx.metricx_scorer import MetricXScorer


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to score GenMT translations with an MT metric specified in input."
    )

    parser.add_argument(
        "--translations-path",
        type=Path,
        required=True,
        help="[REQUIRED] Path to the directory containing the translations to be scored. The directory must contain a "
        "jsonl file for each MT system.",
    )

    parser.add_argument(
        "--refs-path",
        type=Path,
        help="Path to the directory containing the reference translations. The directory must contain one directory for"
        " each language pair with human refs. In each language pair without human references, only QE metrics will be "
        "used.",
    )

    parser.add_argument(
        "--testset-path",
        type=Path,
        required=True,
        help="[REQUIRED] Path to a jsonl file containing the WMT test set. This is used to extract the source texts for"
        " scoring.",
    )

    parser.add_argument(
        "--lps-to-score",
        type=str,
        nargs="+",
        help="If passed, only the translations for these language pairs will be scored. They must be passed in local "
        "code format (e.g., 'en-et_EE').",
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=[
            "gemba-esa",
            "metricx24-hybrid-xl",
            "xcomet-xl",
            "cometkiwi-xl",
            "gemba-mqm",
        ],
        default="gemba-esa",
        help="Which MT metric to use for scoring. Allowed values: 'gemba-esa', 'metricx24-hybrid-xl', 'xcomet-xl', "
        "'cometkiwi-xl', and 'gemba-mqm'. Default: 'gemba-esa'.",
    )

    parser.add_argument(
        "--disk-cache-path",
        type=Path,
        help="Optional path to a disk cache directory for saving scores. If not provided, a default one will be used.",
    )

    parser.add_argument(
        "--gemba-model",
        type=str,
        choices=["gpt-4o", "gpt-4.1", "command-a-03-2025"],
        default="gpt-4.1",
        help="The LLM model to call with API for GEMBA. Default: 'gpt-4.1'.",
    )

    parser.add_argument(
        "--metricx24-predict-script-path",
        type=Path,
        help="Path to the metricx24 predict.py script to use for running inference with metricx. Required if the metric"
        " specified is 'metricx24-hybrid-qe'.",
    )

    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Batch size to use when running inference with the non-gemba metrics. Default: 32.",
    )

    parser.add_argument(
        "--scored-translations-path",
        type=Path,
        required=True,
        help="[REQUIRED] Path to the output pickle file where the scored translations will be saved.",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level. Default: INFO.",
    )

    return parser


def matching_paragraphs_check(
    lp2domain_test_docs: Dict[str, Dict[str, Dict[str, List[str]]]],
    sys2translations: Dict[str, Dict[str, Dict[str, Dict[str, List[str]]]]],
) -> None:
    """
    Check if the number of paragraphs in the translations matches the number of paragraphs in the test set documents.

    Args:
        lp2domain_test_docs: Nested dict for src: lp -> domain -> document_id -> list of paragraphs.
        sys2translations: Nested dict for tgt: sys -> lp -> domain -> document_id -> list of translated paragraphs.

    Raises:
        ValueError: If the number of paragraphs in the translations does not match the number of source paragraphs.
    """
    systems_to_filter = []
    for sys, lp2domain_translated_docs in sys2translations.items():
        for lp, domain2translated_docs in lp2domain_translated_docs.items():
            assert set(domain2translated_docs) == set(lp2domain_test_docs[lp]), (
                f"Mismatch in domains for {sys}, {lp}: {set(domain2translated_docs)} (in translations) vs "
                f"{set(lp2domain_test_docs[lp])} (in test set)."
            )
            for domain, translated_docs in domain2translated_docs.items():
                if set(translated_docs) != set(lp2domain_test_docs[lp][domain]):
                    systems_to_filter.append(sys)
                    break
                for doc_id, translated_paragraphs in translated_docs.items():
                    assert isinstance(translated_paragraphs, list) and all(
                        isinstance(p, str) for p in translated_paragraphs
                    )
                    if len(translated_paragraphs) != len(
                        lp2domain_test_docs[lp][domain][doc_id]
                    ):
                        systems_to_filter.append(sys)
                        break
                if len(systems_to_filter) > 0 and systems_to_filter[-1] == sys:
                    break
            if len(systems_to_filter) > 0 and systems_to_filter[-1] == sys:
                break

    for sys in systems_to_filter:
        del sys2translations[sys]
    logging.info(f"Systems filtered: {systems_to_filter}.")


def main() -> None:
    args: Namespace = read_arguments().parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    lp2refs = dict()  # nested dict: lp -> domain -> document_id -> list of refs
    if args.refs_path is not None:
        for lp in args.refs_path.iterdir():
            if lp.is_dir():
                lp2refs[lp.name] = dict()
                for domain in lp.iterdir():
                    if domain.is_dir():
                        lp2refs[lp.name][domain.name] = dict()
                        for doc in domain.iterdir():
                            if doc.is_file() and doc.suffix == ".txt":
                                doc_id = doc.stem
                                if lp.name == "en-et_EE" and doc_id.endswith("_ET_C"):
                                    doc_id = doc_id[
                                        :-5
                                    ]  # Remove the "_ET_C" suffix present in en-et_EE ref doc files
                                # Read the file content
                                content = doc.read_text(encoding="utf-8")
                                # Split into paragraphs by double-newline
                                paragraphs = [
                                    p for p in content.split("\n\n") if p.strip()
                                ]
                                # Store in nested dictionary
                                lp2refs[lp.name][domain.name][doc_id] = paragraphs
        logging.info(
            f"Loaded reference translations for the following language pairs: {list(lp2refs)}."
        )

    lp2domain_test_docs = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )  # nested dict: lp -> domain -> document_id -> list of paragraphs
    n_test_docs_to_eval = 0
    with open(args.testset_path, "r", encoding="utf-8") as f:
        for line in f:
            test_doc = json.loads(line)

            if test_doc["collection_id"] == "testsuites":
                continue  # Skip instances from the "testsuites" collection for automatic evaluation

            lp, domain, document_id = test_doc["doc_id"].split("_#_")
            if domain == "speech":
                assert document_id.startswith("vid_")
                document_id = document_id[
                    4:
                ]  # Remove the "vid_" prefix from speech document IDs to match the ones in the refs
            lp2domain_test_docs[lp][domain][document_id] = [
                {"src": src} for src in test_doc["src_text"].split("\n\n")
            ]
            if lp in lp2refs:
                if len(lp2domain_test_docs[lp][domain][document_id]) != len(
                    lp2refs[lp][domain][document_id]
                ):
                    raise ValueError(
                        f"Mismatch in number of paragraphs for {lp}, {domain}, {document_id} between source texts "
                        f"({len(lp2domain_test_docs[lp][domain][document_id])}) and references translations ("
                        f"{len(lp2refs[lp][domain][document_id])})."
                    )
                for seg_idx in range(len(lp2domain_test_docs[lp][domain][document_id])):
                    lp2domain_test_docs[lp][domain][document_id][seg_idx][
                        "ref"
                    ] = lp2refs[lp][domain][document_id][seg_idx]

            n_test_docs_to_eval += 1
    logging.info(f"Loaded GenMT test set with {n_test_docs_to_eval} docs to eval.")

    sys2translations = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )  # nested dict: sys -> lp -> domain -> document_id -> list of translated paragraphs
    sys2n_none_translations = defaultdict(int)  # sys -> number of None translations

    with open(args.translations_path / "teams.json", "r", encoding="utf-8") as f:
        teams = json.load(f)
    filename2sys = dict()
    for entry in teams:
        publication_name = entry["publication_name"]
        assert isinstance(publication_name, str) and len(publication_name) > 0
        primary_submissions = entry.get("primary_submissions", [])
        assert len(primary_submissions) == 1
        for sub in primary_submissions:
            assert sub["competition"] == "WMT25: General MT"
            file_name = sub["file_name"]

            assert file_name.startswith("submissions/")
            short_name = file_name[len("submissions/") :]

            last_dot_char_idx = short_name.rfind(".")
            filename2sys[short_name[:last_dot_char_idx]] = publication_name
    assert len(filename2sys.values()) == len(
        set(filename2sys.values())
    ), "There are duplicate publication names in the teams.json file. Please check the file for duplicates."

    for filename in args.translations_path.iterdir():
        if filename.suffix == ".jsonl":
            sys = filename2sys.get(filename.stem)
            if sys is None:
                logging.info(f"No system found for file {filename}. Skipping.")
                continue  # Skip files without a corresponding system

            with filename.open("r", encoding="utf-8") as f:
                translated_docs = [json.loads(line) for line in f]
            for translation_data in translated_docs:
                if translation_data["hypothesis"] is None:
                    sys2n_none_translations[sys] += 1
                    continue  # Skip translations that are None
                lp, domain, document_id = translation_data["doc_id"].split("_#_")
                if domain == "testsuite":
                    continue  # Skip instances from the "testsuites" collection for automatic evaluation
                if domain == "speech":
                    assert document_id.startswith("vid_")
                    document_id = document_id[4:]  # To match the ones in the test set
                sys2translations[sys][lp][domain][document_id] = translation_data[
                    "hypothesis"
                ].split("\n\n")
    logging.info(f"Loaded {len(sys2translations)} MT systems with doc translations.")
    if len(sys2n_none_translations) > 0:
        logging.warning("Some MT systems have None translations:")
        logging.warning(pformat(sys2n_none_translations))

    matching_paragraphs_check(lp2domain_test_docs, sys2translations)

    if args.metric == "gemba-esa" or args.metric == "gemba-mqm":
        scorer = GembaScorer(args.metric.upper(), args.gemba_model)
        sys2seg_outputs = scorer.score(
            lp2domain_test_docs,
            sys2translations,
            disk_cache_path=args.disk_cache_path,
            lps_to_score=args.lps_to_score,
        )
    else:
        scorer = (
            MetricXScorer(args.metricx24_predict_script_path)
            if args.metric == "metricx24-hybrid-xl"
            else CometScorer(args.metric)
        )
        sys2seg_outputs = scorer.score(
            lp2domain_test_docs,
            sys2translations,
            batch_size=args.batch_size,
            disk_cache_path=args.disk_cache_path,
            lps_to_score=args.lps_to_score,
        )

    def recursive_defaultdict_to_dict(d):
        if isinstance(d, defaultdict):
            d = {k: recursive_defaultdict_to_dict(v) for k, v in d.items()}
        elif isinstance(d, dict):
            d = {k: recursive_defaultdict_to_dict(v) for k, v in d.items()}
        return d

    sys2seg_outputs = recursive_defaultdict_to_dict(
        sys2seg_outputs
    )  # To avoid a pickle error with defaultdicts
    logging.info(f"Scored {len(sys2seg_outputs)} MT systems with {args.metric}.")
    args.scored_translations_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.scored_translations_path, "wb") as handle:
        pickle.dump(sys2seg_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Wrote scores to {args.scored_translations_path}")


if __name__ == "__main__":
    main()
