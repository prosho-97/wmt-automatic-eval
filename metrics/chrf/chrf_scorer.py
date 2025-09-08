import logging
from collections import defaultdict
from typing import List, Dict, Optional

import sacrebleu
from tqdm import tqdm


class ChrFScorer:
    ALLOWED_LPS = {"en-bho_IN", "en-mas_KE"}

    def __init__(self):
        self.metric = "chrf++"
        # SacreBLEU's chrF++ is the default ('word_order' = 2 means chrF++)
        self.chrf = sacrebleu.CHRF(word_order=2)

    def get_signature(self, nrefs: int = 1) -> str:
        """
        Get the ChrF++ string signature from the SacreBLEU package.

        Args:
           nrefs: Number of references to consider.
        """
        # Prime: run a dummy evaluation to let the metric know the number of references
        _ = self.chrf.sentence_score("", [""] * nrefs)
        return self.chrf.get_signature()

    def score(
        self,
        lp2domain_test_docs: Dict[str, Dict[str, Dict[str, List[Dict[str, str]]]]],
        sys2translations: Dict[str, Dict[str, Dict[str, Dict[str, List[str]]]]],
        lps_to_score: Optional[List[str]] = None,
    ) -> Dict[
        str,  # sys
        Dict[
            str,  # lp
            Dict[
                str,  # domain
                Dict[
                    str,  # doc_id
                    List[float],  # list of chrF++ scores per paragraph
                ],
            ],
        ],
    ]:
        """
        Compute chrF++ scores for provided translations. Only 'en-bho_IN' and 'en-mas_KE' language pairs are supported.

        Args:
            lp2domain_test_docs: Nested dict for src: lp -> domain -> document_id -> list of paragraphs.
            sys2translations: Nested dict for tgt: sys -> lp -> domain -> document_id -> list of translated paragraphs.
            lps_to_score: Optional language pairs to score. If provided, only these language pairs will be scored.

        Returns:
            sys2seg_outputs: Nested out structure mirroring sys2translations: sys -> lp -> domain -> doc_id -> [scores].
        """
        # Determine which LPs to score
        if lps_to_score is not None:
            valid_lps_to_score = self.ALLOWED_LPS.intersection(lps_to_score)
            if len(valid_lps_to_score) == 0:
                raise ValueError(
                    f"No valid language pairs found in lps_to_score for chrF++: {lps_to_score}. The only valid language"
                    f" pairs to score with chrF++ are the following: {self.ALLOWED_LPS}."
                )
            lps_to_score = valid_lps_to_score
        else:
            lps_to_score = self.ALLOWED_LPS

        sys2seg_outputs = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )  # nested dict: sys -> lp -> domain -> doc_id -> list of paragraph outputs

        for sys, lp2domain_translated_docs in tqdm(
            sys2translations.items(), desc="Systems"
        ):
            for lp, domain2translated_docs in tqdm(
                lp2domain_translated_docs.items(), desc="Language pairs"
            ):
                if lp not in lps_to_score:
                    continue

                for domain, translated_docs in domain2translated_docs.items():
                    for doc_id, translated_paragraphs in translated_docs.items():
                        refs = [
                            seg["ref"]
                            for seg in lp2domain_test_docs[lp][domain][doc_id]
                        ]
                        if len(refs) != len(translated_paragraphs):
                            raise ValueError(
                                f"Mismatch in # of refs ({len(refs)}) and hyps ({len(translated_paragraphs)}) "
                                f"for {sys}, {lp}, {domain}, {doc_id}"
                            )

                        # Compute chrF++ for each segment = paragraph
                        scores = [
                            self.chrf.sentence_score(hyp, [ref]).score
                            for hyp, ref in zip(translated_paragraphs, refs)
                        ]
                        sys2seg_outputs[sys][lp][domain][doc_id] = scores

        return sys2seg_outputs
