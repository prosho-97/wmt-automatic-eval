import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal, List, Dict, Optional

import diskcache as dc

from tqdm import tqdm

import comet


class CometScorer:
    def __init__(self, model: str = Literal["xcomet-xl", "cometkiwi-xl"]):
        """
        Class for scoring translations using a COMET metric (either XCOMET-XL or CometKiwi-XL).

        Args:
            model: The model to use for Comet scoring. Allowed values: "xcomet-xl", "cometkiwi-xl".

        Raises:
            ValueError: If the model is not supported.
        """
        if model != "xcomet-xl" and model != "cometkiwi-xl":
            raise ValueError(
                f"Unsupported model: {model}. Supported models are 'xcomet-xl' and 'cometkiwi-xl'."
            )

        self.model = model

        comet_metric_model_path = comet.download_model(
            "Unbabel/XCOMET-XL"
            if self.model == "xcomet-xl"
            else "Unbabel/wmt23-cometkiwi-da-xl"
        )
        self.comet_metric_model = comet.load_from_checkpoint(comet_metric_model_path)

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
                    List[float],  # list of COMET scores per paragraph
                ],
            ],
        ],
    ]:
        """
        Calculate COMET scores for the provided source texts and translations.

        Args:
            lp2domain_test_docs: Nested dict for src: lp -> domain -> document_id -> list of paragraphs.
            sys2translations: Nested dict for tgt: sys -> lp -> domain -> document_id -> list of translated paragraphs.
            batch_size: Batch size for COMET scoring. Default: 32.
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

        sys2seg_outputs = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )  # nested dict: sys -> lp -> domain -> doc_id -> list of paragraph outputs

        def create_input_data_for_comet_metric_model(
            src: List[str], cand: List[str], ref: Optional[List[str]] = None
        ) -> List[Dict[str, str]]:
            """
            Create the input data for a COMET metric model.

            Args:
                src: Source texts.
                cand: Candidate Translations.
                ref: Reference translations. Default: None.

            Returns:
                List[Dict[str, str]]: Input data for a COMET metric model.

            Raises:
                ValueError: If the lengths of the input lists do not match.
            """
            if len(src) != len(cand):
                raise ValueError(
                    f"The number of source texts ({len(src)}) and candidate translations ({len(cand)}) must be the "
                    "same!"
                )
            if ref is not None and len(src) != len(ref):
                raise ValueError(
                    f"The number of source texts and translations ({len(src)}) does not match the number of references "
                    f"({len(ref)})!"
                )

            return (
                [{"src": s, "mt": c} for s, c in zip(src, cand)]
                if ref is None
                else [{"src": s, "mt": c, "ref": r} for s, c, r in zip(src, cand, ref)]
            )

        if lps_to_score is not None:
            lps_to_score = set(lps_to_score)

        for sys, lp2domain_translated_docs in tqdm(
            sys2translations.items(), desc="Systems"
        ):
            for lp, domain2translated_docs in lp2domain_translated_docs.items():
                if lps_to_score is not None and lp not in lps_to_score:
                    continue

                cache_id = {"sys": sys, "lp": lp}

                meta = []
                lp_source_texts, lp_translations, lp_refs = [], [], []

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
                            if "ref" in seg_data and not self.model.startswith(
                                "cometkiwi"
                            ):
                                lp_refs.append(seg_data["ref"])
                            meta.append((domain, doc_id, seg_idx))

                if cache_id in cache:
                    scores, meta = (
                        cache[cache_id]["scores"],
                        cache[cache_id]["meta"],
                    )
                    logging.info(
                        f"Using cached scores for {sys}, {lp} from disk cache for COMET."
                    )
                else:
                    logging.info(
                        f"Calculating scores for {sys}, {lp} and saving to disk cache for COMET."
                    )
                    scores = self.comet_metric_model.predict(
                        create_input_data_for_comet_metric_model(
                            lp_source_texts,
                            lp_translations,
                            lp_refs if len(lp_refs) > 0 else None,
                        ),
                        batch_size=batch_size,
                        gpus=1,
                    ).scores
                    cache[cache_id] = {"scores": scores, "meta": meta}
                assert len(scores) == len(meta), (
                    f"Mismatch in number of scores ({len(scores)}) "
                    f"and meta entries ({len(meta)}) for {sys}, {lp}."
                )

                # Prebuild structure: sys → lp → domain → doc_id → [None]*N
                for domain, translated_docs in domain2translated_docs.items():
                    for doc_id, translated_paragraphs in translated_docs.items():
                        sys2seg_outputs[sys][lp][domain][doc_id] = [None] * len(
                            translated_paragraphs
                        )

                # Fill per-paragraph
                for (domain, doc_id, seg_idx), score in zip(meta, scores):
                    sys2seg_outputs[sys][lp][domain][doc_id][seg_idx] = score

        return sys2seg_outputs
