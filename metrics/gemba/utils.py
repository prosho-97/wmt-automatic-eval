import ipdb
import pandas as pd
import diskcache as dc
from metrics.gemba.gpt_api import GptApi
from metrics.gemba.gemba_mqm_utils import (
    TEMPLATE_GEMBA_MQM,
    apply_template,
    parse_mqm_answer,
)
from metrics.gemba.gemba_esa_utils import (
    TEMPLATE_GEMBA_ESA_ERROR_SPANS,
    TEMPLATE_GEMBA_ESA_RANKING,
)
from metrics.gemba.prompt import prompts, validate_number


def get_gemba_scores(source, hypothesis, source_lang, target_lang, method, model):
    df = pd.DataFrame({"source_seg": source, "target_seg": hypothesis})
    df["source_lang"] = source_lang
    df["target_lang"] = target_lang

    cache = dc.Cache(
        f"cache/{model}_{method}",
        expire=None,
        size_limit=int(10e10),
        cull_limit=0,
        eviction_policy="none",
    )
    gptapi = GptApi()

    if method == "GEMBA-MQM":
        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_MQM, x), axis=1)
        parse_answer = lambda x: parse_mqm_answer(
            x, list_mqm_errors=False, full_desc=True
        )
        answers = gptapi.bulk_request(
            df, model, parse_answer, cache=cache, max_tokens=500
        )
    elif method == "GEMBA-ESA":
        df["prompt"] = df.apply(
            lambda x: apply_template(TEMPLATE_GEMBA_ESA_ERROR_SPANS, x), axis=1
        )
        parse_answer = lambda x: x
        error_spans = gptapi.bulk_request(df, model, parse_answer, cache=cache)
        df["error_spans"] = pd.DataFrame(error_spans)["answer"]

        df["prompt"] = df.apply(
            lambda x: apply_template(TEMPLATE_GEMBA_ESA_RANKING, x), axis=1
        )
        parse_answer = validate_number
        answers = gptapi.bulk_request(df, model, parse_answer, cache=cache)
    else:
        raise Exception(f"Method {method} not supported.")

    return list(pd.DataFrame(answers)["answer"])
