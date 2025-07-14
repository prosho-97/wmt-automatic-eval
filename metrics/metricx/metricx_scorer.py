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
        refs: List[Optional[str]],
        tmp_path: Path,
        batch_size: int = 32,
        is_qe: bool = True,
    ) -> List[float]:
        """
        Launch the MetricX scoring script.

        Args:
            source_texts: Source texts list.
            translations: Translations list.
            refs: References list. May contain None values if references are not available.
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
        lps_to_score: Optional[List[str]] = None,
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
            lps_to_score: Optional language pairs to score. If provided, only these language pairs will be scored.

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

        if lps_to_score is not None:
            lps_to_score = set(lps_to_score)

        try:
            for sys, lp2domain_translated_docs in tqdm(
                sys2translations.items(), desc="Systems"
            ):
                for lp, domain2translated_docs in tqdm(
                    lp2domain_translated_docs.items(), "Language pairs"
                ):
                    if lps_to_score is not None and lp not in lps_to_score:
                        continue

                    # To preserve the output structure, pre-fill outputs with None
                    for domain, translated_docs in domain2translated_docs.items():
                        for doc_id, translated_paragraphs in translated_docs.items():
                            sys2seg_outputs[sys][lp][domain][doc_id] = [None] * len(
                                translated_paragraphs
                            )

                    is_qe = True

                    # Collect details per segment
                    all_segments = []
                    for domain, translated_docs in domain2translated_docs.items():
                        for doc_id, translated_paragraphs in translated_docs.items():
                            src_list = lp2domain_test_docs[lp][domain][doc_id]
                            assert len(src_list) == len(
                                translated_paragraphs
                            ), f"Mismatch in {sys}, {lp}, {domain}, {doc_id}"
                            for seg_idx, (seg_data, tgt) in enumerate(
                                zip(src_list, translated_paragraphs)
                            ):
                                src = seg_data["src"]
                                ref = seg_data.get("ref")
                                if ref is not None:
                                    is_qe = False
                                # cache ref=None segments if reference is not available.
                                cache_id: Dict[str, Optional[str]] = {
                                    "src": src,
                                    "mt": tgt,
                                    "ref": ref,
                                }
                                all_segments.append(
                                    (domain, doc_id, seg_idx, cache_id, src, tgt, ref)
                                )

                    uncached_inputs = []
                    uncached_meta = []  # list of (domain, doc_id, seg_idx, cache_id)
                    cached_scores = dict()

                    for meta in all_segments:
                        domain, doc_id, seg_idx, cache_id, src, mt, ref = meta
                        cache_key_tuple = tuple(
                            sorted(cache_id.items())
                        )  # Diskcache only supports hashable keys. Sorting to ensure consistency.
                        if cache_key_tuple in cache:
                            cached_scores[(domain, doc_id, seg_idx)] = cache[
                                cache_key_tuple
                            ]
                        else:
                            uncached_inputs.append((src, mt, ref))
                            uncached_meta.append(
                                (domain, doc_id, seg_idx, cache_key_tuple)
                            )

                    # Score and update the cache for missing entries
                    if uncached_inputs:
                        logging.info(
                            f"Calculating {len(uncached_inputs)} scores for {sys}, {lp} and saving to disk cache for "
                            "MetricX."
                        )
                        srcs, mts, refs = [], [], []
                        for src, mt, ref in uncached_inputs:
                            srcs.append(src)
                            mts.append(mt)
                            refs.append(ref)
                        if refs[0] is not None:
                            logging.info("Scoring with references.")

                        scores = self.launch_script(
                            srcs,
                            mts,
                            refs,
                            tmp_path,
                            batch_size,
                            is_qe,
                        )
                        assert len(scores) == len(uncached_meta), (
                            f"Mismatch in number of scores ({len(scores)}) "
                            f"and meta entries ({len(uncached_meta)}) for {sys}, {lp}."
                        )
                        # Fill cache, indexed by the relevant cache tuple
                        for (domain, doc_id, seg_idx, cache_key_tuple), score in zip(
                            uncached_meta, scores
                        ):
                            cache[cache_key_tuple] = score
                            cached_scores[(domain, doc_id, seg_idx)] = score

                    logging.info(
                        f"Scores complete for {sys}, {lp} (total segments: {len(all_segments)})."
                    )

                    # Write output scores into result structure
                    for domain, doc_id, seg_idx, cache_id, src, mt, ref in all_segments:
                        score = cached_scores[(domain, doc_id, seg_idx)]
                        sys2seg_outputs[sys][lp][domain][doc_id][seg_idx] = score

        finally:
            # Clean up the temporary file
            if tmp_path.exists():
                tmp_path.unlink()

        return sys2seg_outputs
