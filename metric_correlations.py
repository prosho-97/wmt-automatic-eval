import pandas as pd
import json
from matplotlib import cm, colors as mcolors
import ipdb
import numpy as np

from textwrap import dedent


def df_to_latex_heatmap(df: pd.DataFrame, cmap_name="RdYlGn", fmt="{:.3f}"):
    import numpy as np
    import pandas as pd
    from matplotlib import cm, colors as mcolors

    vmin = np.nanmin(df.values.astype(float))
    vmax = np.nanmax(df.values.astype(float))
    denom = (vmax - vmin) if vmax != vmin else 1.0

    cmap = cm.get_cmap(cmap_name)

    def hex_and_text(x):
        if pd.isna(x):
            return ""
        g = (float(x) - vmin) / denom
        rgb = cmap(g)[:3]
        hexcol = mcolors.to_hex(rgb, keep_alpha=False).lstrip("#").upper()
        r, g_, b = int(hexcol[0:2],16), int(hexcol[2:4],16), int(hexcol[4:6],16)
        lum = 0.299*r + 0.587*g_ + 0.114*b
        txt = "000000" if lum > 128 else "FFFFFF"
        return rf"\cellcolor[HTML]{{{hexcol}}}\textcolor[HTML]{{{txt}}}{{{fmt.format(x)}}}"

    tex_df = df.copy()
    for c in tex_df.columns:
        tex_df[c] = tex_df[c].map(hex_and_text)

    def escape_latex(s):
        if not isinstance(s, str):
            return s
        return (s.replace("\\", r"\textbackslash{}")
                 .replace("_", r"\_")
                 .replace("&", r"\&")
                 .replace("%", r"\%")
                 .replace("$", r"\$")
                 .replace("#", r"\#")
                 .replace("{", r"\{")
                 .replace("}", r"\}")
                 .replace("~", r"\textasciitilde{}")
                 .replace("^", r"\textasciicircum{}"))

    tex_df.index = [escape_latex(i) for i in df.index]

    # Modify column names to be two lines
    def multiline_header(col_name):
        if " - " in col_name:
            part1, part2 = col_name.split(" - ", 1)
            part1 = escape_latex(part1)
            part2 = escape_latex(part2)
            return rf"\shortstack{{{part1}\\{part2}}}"
        else:
            return escape_latex(col_name)

    tex_df.columns = [multiline_header(c) for c in df.columns]

    return tex_df.to_latex(escape=False)




#  load metric_correlations_per_language_pair.json
with open('metric_correlations_per_language_pair.json') as f:
    data = json.load(f)

metric_map = {
    "CometKiwi-XL": "Kiwi",
    "GEMBA-ESA-CMDA": "G-CmdA",
    "GEMBA-ESA-GPT4.1": "G-GPT",
    "MetricX-24-Hybrid-XL": "MetX",
    "XCOMET-XL": "xComet"
}


rows = {}
for lang_pair in data:
    row = {}
    for p in data[lang_pair]["pairs"]:
        col_name = f"{metric_map[p['metric_i']]} - {metric_map[p['metric_j']]}"
        row[col_name] = p['r']  # store Pearson correlation
    rows[lang_pair] = row 

# Create DataFrame
df = pd.DataFrame.from_dict(rows, orient="index")


latex = df_to_latex_heatmap(df)

# (optional) save to file
with open("metrics_correlations.tex", "w") as f:
    f.write(latex)

