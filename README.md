# WMT25 Automatic Evaluation

All the automatic metric scores have already been computed, and they can be found in [data/metrics_outputs](data/metrics_outputs).

---

## Data setup

Clone inside the `data` folder the `wmt25-general-mt` repository containing MT system outputs and human references:

```bash
cd data
git clone https://github.com/wmt-conference/wmt25-general-mt
```

---

## Automatic evaluation

<details>
<summary><strong>1) GEMBA-ESA</strong></summary>

<br/>

To score with GEMBA-ESA, you need to set the required API keys in your environment (Cohere or OpenAI).

<br/>

<details>
<summary><strong>1.1) GEMBA-ESA-CmdA</strong></summary>

```bash
python main.py \
        --translations-path data/wmt25-general-mt/data \
        --testset-path data/wmt25-general-mt/data/wmt25-genmt.jsonl \
        --metric gemba-esa \
        --gemba-model command-a-03-2025 \
        --lps-to-score <LP> <LP> ... \ # e.g. cs-de_DE cs-uk_UA en-cs_CZ. If not specified, the scoring will run for all language pairs.
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
        --lps-to-score <LP> <LP> ... \ # e.g. cs-de_DE cs-uk_UA en-cs_CZ. If not specified, the scoring will run for all language pairs.
        --scored-translations-path data/metrics_outputs/GEMBA-ESA-GPT4.1/outputs.pickle \
```

</details>

</details>

<details>
<summary><strong>2) MetricX-24-Hybrid-XL</strong></summary>

After cloning and installing the official MetricX repository, note that the `predict.py` script currently does not support `--batch-size > 1` ([issue #2](https://github.com/google-research/metricx/issues/2)). If you need a batch size greater than 1, clone and install the fork at [prosho-97/metricx](https://github.com/prosho-97/metricx) using the `new_requirements.txt` file.
For language pairs with human reference translations available, [MetricX-24-Hybrid-XL](https://huggingface.co/google/metricx-24-hybrid-xl-v2p6) will score in reference-based mode, otherwise in QE mode.
 
```bash
python main.py \
        --translations-path data/wmt25-general-mt/data
        --testset-path data/wmt25-general-mt/data/wmt25-genmt.jsonl
        --lps-to-score <LP> <LP> ... \ # e.g. cs-de_DE cs-uk_UA en-cs_CZ. If not specified, the scoring will run for all language pairs.
        --metric metricx24-hybrid-xl
        --metricx24-predict-script-path <METRICX preict.py SCRIPT PATH>
        --batch-size 8
        --scored-translations-path data/metrics_outputs/MetricX-24-Hybrid-XL/outputs.pickle
```

</details>

<details>
<summary><strong>3) XCOMET-XL</strong></summary>

As with MetricX-24-Hybrid-XL above, for language pairs with human reference translations available, [XCOMET-XL](https://huggingface.co/Unbabel/XCOMET-XL) will score in reference-based mode, otherwise in QE mode.

```bash
python main.py \
        --translations-path data/wmt25-general-mt/data
        --testset-path data/wmt25-general-mt/data/wmt25-genmt.jsonl
        --lps-to-score <LP> <LP> ... \ # e.g. cs-de_DE cs-uk_UA en-cs_CZ. If not specified, the scoring will run for all language pairs.
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
        --lps-to-score <LP> <LP> ... \ # e.g. cs-de_DE cs-uk_UA en-cs_CZ. If not specified, the scoring will run for all language pairs.
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
