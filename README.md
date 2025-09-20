# WMT25 Automatic Evaluation

All the automatic metric scores have already been computed, and they can be found in [data/metrics_outputs](data/metrics_outputs).

---

## Data Setup

Clone inside the `data` folder the `wmt25-general-mt` repository containing MT system outputs and human references:

```bash
cd data
git clone https://github.com/wmt-conference/wmt25-general-mt
```

---

## Compute Metric Scores

In all the commands below, `--lps-to-score` accepts a space-separated list of language pairs, e.g., `cs-de_DE cs-uk_UA en-cs_CZ`. If omitted, all WMT25 language pairs are scored. The only exception is `chrF++`, which can run only on `en-bho_IN` and `en-mas_KE`.

<details>
<summary><strong>1) GEMBA-ESA</strong></summary>

<br/>

To score with `GEMBA-ESA`, you need to set the required API keys in your environment (Cohere or OpenAI).

<br/>

<details>
<summary><strong>1.1) GEMBA-ESA-CmdA</strong></summary>

```bash
python main.py \
        --translations-path data/wmt25-general-mt/data \
        --testset-path data/wmt25-general-mt/data/wmt25-genmt.jsonl \
        --metric gemba-esa \
        --gemba-model command-a-03-2025 \
        --lps-to-score cs-de_DE cs-uk_UA en-cs_CZ
        --scored-translations-path data/metrics_outputs/GEMBA-ESA-CMDA/outputs.pickle
```

</details>

<details>
<summary><strong>1.2) GEMBA-ESA-GPT-4.1</strong></summary>

```bash
python main.py \
        --translations-path data/wmt25-general-mt/data \
        --testset-path data/wmt25-general-mt/data/wmt25-genmt.jsonl \
        --metric gemba-esa \
        --gemba-model gpt-4.1 \
        --lps-to-score cs-de_DE cs-uk_UA en-cs_CZ
        --scored-translations-path data/metrics_outputs/GEMBA-ESA-GPT4.1/outputs.pickle \
```

</details>

With `GEMBA-ESA`, there is the option of scoring human reference translations for the language pairs where they are available, which are the followoing: `cs-de_DE cs-uk_UA en-ar_EG en-bho_IN en-cs_CZ en-et_EE en-is_IS en-ja_JP en-ko_KR en-mas_KE en-ru_RU en-sr_Cyrl_RS en-uk_UA en-zh_CN ja-zh_CN`. To do this, add the `--score-only-refs` flag to the commands above and set the output pickle filename to `outputs_refA.pickle`.

</details>

<details>
<summary><strong>2) MetricX-24-Hybrid-XL</strong></summary>

After cloning and installing the official MetricX repository, note that the `predict.py` script currently does not support `--batch-size > 1` ([issue #2](https://github.com/google-research/metricx/issues/2)). If you need a batch size greater than 1, clone and install the fork at [prosho-97/metricx](https://github.com/prosho-97/metricx) using the `new_requirements.txt` file.
For language pairs with human reference translations available, [MetricX-24-Hybrid-XL](https://huggingface.co/google/metricx-24-hybrid-xl-v2p6) will score in reference-based mode, otherwise in QE mode.
 
```bash
python main.py \
        --translations-path data/wmt25-general-mt/data
        --testset-path data/wmt25-general-mt/data/wmt25-genmt.jsonl
        --lps-to-score cs-de_DE cs-uk_UA en-cs_CZ
        --metric metricx24-hybrid-xl
        --metricx24-predict-script-path <METRICX predict.py SCRIPT PATH>
        --batch-size 8
        --scored-translations-path data/metrics_outputs/MetricX-24-Hybrid-XL/outputs.pickle
```

</details>

<details>
<summary><strong>3) XCOMET-XL</strong></summary>

As with `MetricX-24-Hybrid-XL` above, for language pairs with human reference translations available, [XCOMET-XL](https://huggingface.co/Unbabel/XCOMET-XL) will score in reference-based mode, otherwise in QE mode.

```bash
python main.py \
        --translations-path data/wmt25-general-mt/data
        --testset-path data/wmt25-general-mt/data/wmt25-genmt.jsonl
        --lps-to-score cs-de_DE cs-uk_UA en-cs_CZ
        --metric xcomet-xl
        --batch-size 8
        --scored-translations-path data/metrics_outputs/XCOMET-XL/outputs.pickle
```

</details>

<details>
<summary><strong>4) CometKiwi-XL</strong></summary>

```bash
python main.py \
        --translations-path data/wmt25-general-mt/data
        --testset-path data/wmt25-general-mt/data/wmt25-genmt.jsonl
        --lps-to-score cs-de_DE cs-uk_UA en-cs_CZ.
        --metric cometkiwi-xl
        --batch-size 8
        --scored-translations-path data/metrics_outputs/CometKiwi-XL/outputs.pickle
```

</details>

<details>
<summary><strong>5) chrF++</strong></summary>

```bash
python main.py \
        --translations-path data/wmt25-general-mt/data
        --testset-path data/wmt25-general-mt/data/wmt25-genmt.jsonl
        --lps-to-score en-bho_IN en-mas_KE
        --metric chrf++
        --scored-translations-path data/metrics_outputs/chrF++/outputs.pickle
```

</details>

---

## Final AutoRank

Once you have computed the scores with all the metrics above (saved under [data/metrics_outputs](data/metrics_outputs)), you can finally run **AutoRank** with:

```bash
cd autorank
python compute_autorank.py \
        --lps cs-de_DE cs-uk_UA en-cs_CZ
        --out-path ./res
```

Where `--lps` accepts a space-separated list of language pairs, e.g., `cs-de_DE cs-uk_UA en-cs_CZ`. Default is the empty list. You can optionally retrict the `AutoRank` to one or more domain(s) using the `--domains` input argument. For example: `--domains speech`.
