import json
import logging
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict

import diskcache as dc

from tqdm import tqdm


class MetricXScorer:
    def __init__(
        self,
        metricx_predict_script_path: Path,
        tokenizer: str = "google/mt5-xl",
        model: str = "google/metricx-24-hybrid-xl-v2p6",
        max_input_length: int = 1536,
    ):
        """
        Class for scoring translations using MetricX.

        Args:
            metricx_predict_script_path: Path to the metricx24 predict.py script.
            tokenizer: Tokenizer to use for MetricX. Default: "google/mt5-xl".
            model: Model to use for MetricX. Default: "google/metricx-24-hybrid-xl-v2p6".
            max_input_length: Maximum input length for the MetricX. Default: 1536.
        """
        self.metricx_predict_script_path = metricx_predict_script_path
        self.tokenizer = tokenizer
        self.model = model
        self.max_input_length = max_input_length

    def launch_script(
        self,
        source_texts: List[str],
        translations: List[str],
        refs: List[str],
        tmp_path: Path,
        batch_size: int = 32,
        is_qe: bool = True,
    ) -> List[float]:
        """
        Launch the MetricX scoring script.

        Args:
            source_texts: Source texts list.
            translations: Translations list.
            refs: References list.
            tmp_path: Temporary path to write necessary input MetricX files.
            batch_size: Batch size for MetricX scoring. Default: 32.
            is_qe: If True ===> MetricX-24-Hyrbrid-QE.

        Returns:
            List of MetricX scores (the higher the score, the better the translation).

        Raises:
            ValueError: If the lengths of the input lists do not match.
        """
        if not (len(source_texts) == len(translations) == len(refs)):
            raise ValueError(
                f"Number of source texts ({len(source_texts)}), number of translations ({len(translations)}), and "
                f"number of references ({len(refs)}) must match!"
            )
        # Compose input jsonl content
        input_lines = []
        for src, hyp, ref in zip(source_texts, translations, refs):
            input_lines.append(
                json.dumps(
                    {
                        "source": src,
                        "hypothesis": hyp,
                        "reference": "" if is_qe else ref,
                    },
                    ensure_ascii=False,
                )
            )

        # Write to temporary file for MetricX input
        with tmp_path.open("w", encoding="utf-8") as f:
            for line in input_lines:
                f.write(line + "\n")

        # Call metricx24 script
        command = [
            "python",
            str(self.metricx_predict_script_path),
            "--tokenizer",
            self.tokenizer,
            "--model_name_or_path",
            self.model,
            "--max_input_length",
            str(self.max_input_length),
            "--batch_size",
            str(batch_size),
            "--input_file",
            str(tmp_path),
            "--output_file",
            str(tmp_path),
        ]
        if is_qe:
            command.append("--qe")

        subprocess.run(command, check=True)

        # Read output and update translations
        with tmp_path.open("r", encoding="utf-8") as f:
            predictions = [
                -json.loads(line)["prediction"] for line in f
            ]  # MetricX returns error scores
            assert len(predictions) == len(source_texts)
            return predictions

    def score(
        self,
        lp2domain_test_docs: Dict[str, Dict[str, Dict[str, List[Dict[str, str]]]]],
        sys2translations: Dict[str, Dict[str, Dict[str, Dict[str, List[str]]]]],
        batch_size: int = 32,
        disk_cache_path: Optional[Path] = None,
    ) -> Dict[
        str,  # sys
        Dict[
            str,  # lp
            Dict[
                str,  # domain
                Dict[
                    str,  # doc_id
                    List[float],  # list of MetricX scores per paragraph
                ],
            ],
        ],
    ]:
        """
        Calculate MetricX scores for the provided source texts and translations.

        Args:
            lp2domain_test_docs: Nested dict for src: lp -> domain -> document_id -> list of paragraphs.
            sys2translations: Nested dict for tgt: sys -> lp -> domain -> document_id -> list of translated paragraphs.
            batch_size: Batch size for MetricX scoring. Default: 32.
            disk_cache_path: Optional path to a disk cache directory. If not provided, a default one will be used.

        Returns:
            sys2seg_outputs: Nested out structure mirroring sys2translations: sys -> lp -> domain -> doc_id -> [scores].
        """
        cache = dc.Cache(
            disk_cache_path if disk_cache_path is not None else f"cache/{self.model}",
            expire=None,
            size_limit=int(10e10),
            cull_limit=0,
            eviction_policy="none",
        )

        # Temporary file used as necessary input for MetricX (overwritten each time)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            tmp_path = Path(tmp.name)

        sys2seg_outputs = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )  # nested dict: sys -> lp -> domain -> doc_id -> list of paragraph outputs

        try:
            for sys, lp2domain_translated_docs in tqdm(
                sys2translations.items(), desc="Systems"
            ):
                for lp, domain2translated_docs in lp2domain_translated_docs.items():
                    cache_id = {"sys": sys, "lp": lp}

                    meta = []
                    lp_source_texts, lp_translations, lp_refs = [], [], []

                    is_qe = True

                    for domain, translated_docs in domain2translated_docs.items():
                        for doc_id, translated_paragraphs in translated_docs.items():
                            assert len(lp2domain_test_docs[lp][domain][doc_id]) == len(
                                translated_paragraphs
                            ), f"Mismatch in {sys}, {lp}, {domain}, {doc_id}"
                            for seg_idx, (seg_data, tgt) in enumerate(
                                zip(
                                    lp2domain_test_docs[lp][domain][doc_id],
                                    translated_paragraphs,
                                )
                            ):
                                lp_source_texts.append(seg_data["src"])
                                lp_translations.append(tgt)
                                lp_refs.append(seg_data.get("ref", ""))
                                if "ref" in seg_data:
                                    is_qe = False
                                meta.append((domain, doc_id, seg_idx))

                    if cache_id in cache:
                        scores, meta = (
                            cache[cache_id]["scores"],
                            cache[cache_id]["meta"],
                        )
                        logging.info(
                            f"Using cached scores for {sys}, {lp} from disk cache for MetricX."
                        )
                    else:
                        logging.info(
                            f"Calculating scores for {sys}, {lp} and saving to disk cache for MetricX."
                        )
                        scores = self.launch_script(
                            lp_source_texts,
                            lp_translations,
                            lp_refs,
                            tmp_path,
                            batch_size,
                            is_qe,
                        )
                        cache[cache_id] = {"scores": scores, "meta": meta}
                    assert len(scores) == len(meta), (
                        f"Mismatch in number of scores ({len(scores)}) "
                        f"and meta entries ({len(meta)}) for {sys}, {lp}."
                    )

                    # Now distribute model outputs into structure: sys → lp → domain → doc_id → list[output]
                    # First, prebuild empty lists:
                    for domain, translated_docs in domain2translated_docs.items():
                        for doc_id, translated_paragraphs in translated_docs.items():
                            sys2seg_outputs[sys][lp][domain][doc_id] = [None] * len(
                                translated_paragraphs
                            )

                    # Fill per-paragraph
                    for (domain, doc_id, seg_idx), score in zip(meta, scores):
                        sys2seg_outputs[sys][lp][domain][doc_id][seg_idx] = score

        finally:
            # Clean up the temporary file
            if tmp_path.exists():
                tmp_path.unlink()

        return sys2seg_outputs
