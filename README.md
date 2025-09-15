# wmt-automatic-eval

All the automatic metric scores have already been computed, and they can be found in [data/metrics_outputs](data/metrics_outputs)

---

Commands to run for automatic evaluation:

Clone inside the `data` folder the `wmt25-general-mt` repository containing MT system outputs and human references:

```bash
cd data
git clone https://github.com/wmt-conference/wmt25-general-mt
```

Then, to run automatic evaluation based on GEMBA-ESA-CMDA for all the language pairs with human evaluation:
```bash
python main.py --translations-path data/wmt25-general-mt/data --testset-path data/wmt25-general-mt/data/wmt25-genmt.jsonl --metric gemba-esa --gemba-model command-a-03-2025 --lps-to-score cs-de_DE cs-uk_UA en-ar_EG en-bho_IN en-cs_CZ en-et_EE en-is_IS en-ja_JP en-ko_KR en-mas_KE en-ru_RU en-sr_Cyrl_RS en-uk_UA en-zh_CN ja-zh_CN --scored-translations-path data/metrics_outputs/GEMBA-ESA-CMDA/outputs.pickle
```

To run the automatic evaluation with another metric, change the above `--metric` input argument value.
