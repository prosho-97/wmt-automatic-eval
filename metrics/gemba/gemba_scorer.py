import logging
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple, Literal

import diskcache as dc
import pandas as pd

from tqdm import tqdm

from metrics.gemba.cohere_api import CohereApi
from metrics.gemba.gemba_esa_utils import (
    TEMPLATE_GEMBA_ESA_ERROR_SPANS,
    TEMPLATE_GEMBA_ESA_RANKING,
)
from metrics.gemba.gemba_mqm_utils import (
    apply_template,
    TEMPLATE_GEMBA_MQM,
    parse_mqm_answer,
    apply_template_for_command_a,
)
from metrics.gemba.gpt_api import GptApi
from metrics.gemba.langs import LANG_CODE_2_LANG_COUNTRY
from metrics.gemba.prompt import validate_number


class GembaScorer:
    def __init__(
        self,
        gemba_type: str = "GEMBA-ESA",
        model: Literal["gpt-4o", "gpt-4.1", "command-a-03-2025"] = "gpt-4o",
    ):
        """
        Class for scoring translations using the GEMBA metric (either GEMBA-ESA or GEMBA-MQM).

        Args:
            gemba_type: Type of GEMBA metric to use. Either "GEMBA-ESA" or "GEMBA-MQM". Default: "GEMBA-ESA".
            model: The LLM model to call. Allowed values: "gpt-4o", "gpt-4.1", "command-a-03-2025". Default: "gpt-4o".

        Raises:
            ValueError: If `gemba_type` or `model` are not supported.
        """
        if gemba_type != "GEMBA-ESA" and gemba_type != "GEMBA-MQM":
            raise ValueError(
                f"Unsupported GEMBA type: {gemba_type}. Supported types are 'GEMBA-ESA' and 'GEMBA-MQM'."
            )

        if model != "gpt-4o" and model != "gpt-4.1" and model != "command-a-03-2025":
            raise ValueError(
                f"Unsupported model: {model}. Supported models are 'gpt-4o', 'gpt-4.1', and 'command-a-03-2025'."
            )

        self.gemba_type = gemba_type
        self.model = model

        self.api, self.template_funct = (
            (CohereApi(), apply_template_for_command_a)
            if self.model == "command-a-03-2025"
            else (GptApi(), apply_template)
        )

    def score(
        self,
        lp2domain_test_docs: Dict[str, Dict[str, Dict[str, List[Dict[str, str]]]]],
        sys2translations: Dict[str, Dict[str, Dict[str, Dict[str, List[str]]]]],
        provide_explanations: bool = False,
        score_only_refs: bool = False,
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
                    Union[
                        List[Optional[float]], List[Optional[Tuple[int, str]]]
                    ],  # list of model outputs
                ],
            ],
        ],
    ]:
        """
        Calculate GEMBA scores for the provided source texts and translations.

        Args:
            lp2domain_test_docs: Nested dict for src: lp -> domain -> document_id -> list of paragraphs.
            sys2translations: Nested dict for tgt: sys -> lp -> domain -> document_id -> list of translated paragraphs.
            provide_explanations: If True, returns explanations for the scores. Default: False.
            score_only_refs: If True, only the reference translations will be scored. Default: False.
            disk_cache_path: Optional path to a disk cache directory. If not provided, a default one will be used.
            lps_to_score: Optional language pairs to score. If provided, only these language pairs will be scored.

        Returns:
            sys2seg_outputs: Nested out structure mirroring sys2translations: sys -> lp -> domain -> doc_id -> [scores].

        """
        cache = dc.Cache(
            disk_cache_path
            if disk_cache_path is not None
            else f"cache/{self.gemba_type}_{self.model}",
            expire=None,
            size_limit=int(10e10),
            cull_limit=0,
            eviction_policy="none",
        )

        sys2seg_outputs = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )  # nested dict: sys -> lp -> domain -> doc_id -> list of paragraph outputs

        if lps_to_score is not None:
            lps_to_score = set(lps_to_score)

        for sys, lp2domain_translated_docs in tqdm(
            sys2translations.items(), desc="Systems"
        ):
            sys = "refA" if score_only_refs else sys
            for lp, domain2translated_docs in tqdm(
                lp2domain_translated_docs.items(), "Language pairs"
            ):
                if lps_to_score is not None and lp not in lps_to_score:
                    continue

                src_lang_code, tgt_lang_code = lp.split("-")
                source_lang, target_lang = (
                    LANG_CODE_2_LANG_COUNTRY[src_lang_code]["language"],
                    LANG_CODE_2_LANG_COUNTRY[tgt_lang_code]["language"],
                )

                lp_source_texts, lp_translations = (
                    [],
                    [],
                )  # Prepare source texts and translations for GEMBA
                meta = []  # Will store (domain, doc_id, seg_idx) for each seg

                for domain, translated_docs in domain2translated_docs.items():
                    for (
                        doc_id,
                        translated_paragraphs,
                    ) in translated_docs.items():
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
                            tgt = seg_data["ref"] if score_only_refs else tgt
                            if tgt is None:
                                raise ValueError(
                                    f"Translation for {sys}, {lp}, {domain}, {doc_id}, segment {seg_idx} is None."
                                )
                            elif tgt == "":
                                logging.warning(
                                    f"Empty translation for {sys}, {lp}, {domain}, {doc_id}, segment {seg_idx}."
                                )
                            lp_translations.append(tgt)
                            meta.append((domain, doc_id, seg_idx))

                df = pd.DataFrame(
                    {"source_seg": lp_source_texts, "target_seg": lp_translations}
                )
                df["source_lang"] = source_lang
                df["target_lang"] = target_lang

                if self.gemba_type == "GEMBA-ESA":
                    df["prompt"] = df.apply(
                        lambda x: self.template_funct(
                            TEMPLATE_GEMBA_ESA_ERROR_SPANS, x
                        ),
                        axis=1,
                    )
                    parse_answer = lambda x: x
                    error_spans = self.api.bulk_request(
                        df, self.model, parse_answer, cache=cache, max_tokens=8192
                    )
                    df["error_spans"] = [output["answer"] for output in error_spans]

                    df["prompt"] = df.apply(
                        lambda x: self.template_funct(TEMPLATE_GEMBA_ESA_RANKING, x),
                        axis=1,
                    )
                    parse_answer = partial(
                        validate_number,
                        final_score=True,
                        provide_explanations=provide_explanations,
                    )
                    answers = self.api.bulk_request(
                        df, self.model, parse_answer, cache=cache, max_tokens=8192
                    )
                else:  # GEMBA-MQM (THIS PART OF THE CODE IS NOT TESTED!)
                    df["prompt"] = df.apply(
                        lambda x: self.template_funct(TEMPLATE_GEMBA_MQM, x), axis=1
                    )
                    # In the MQM case, the explanations will just be the list of error spans.
                    parse_answer = lambda x: parse_mqm_answer(
                        x, list_mqm_errors=provide_explanations, full_desc=True
                    )
                    answers = self.api.bulk_request(
                        df, self.model, parse_answer, cache=cache, max_tokens=8196
                    )
                model_outputs = [output["answer"] for output in answers]
                if len(model_outputs) != len(meta):
                    print(
                        f"Mismatch in number of model outputs ({len(model_outputs)}) and meta entries ({len(meta)}) "
                        f"for {sys}, {lp}."
                    )

                # Now distribute model outputs into structure: sys → lp → domain → doc_id → list[output]
                # First, prebuild empty lists:
                for domain, translated_docs in domain2translated_docs.items():
                    for doc_id, translated_paragraphs in translated_docs.items():
                        sys2seg_outputs[sys][lp][domain][doc_id] = [None] * len(
                            translated_paragraphs
                        )

                # Fill per-paragraph
                for (domain, doc_id, seg_idx), output in zip(meta, model_outputs):
                    sys2seg_outputs[sys][lp][domain][doc_id][seg_idx] = output

        return sys2seg_outputs
