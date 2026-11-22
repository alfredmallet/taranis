"""The music page's two extra colormaps (plans-webgpu/MUSIC_PLAN.md, fix round).

    python webgpu/gen_musiccmap.py      # prints the tables for webgpu/music.js (MUSIC_CMAPS)
                                        # and writes devtools/fixtures/music_cmap.json

Both are in the form every colormap of the demo takes: a degree-6 polynomial per channel
in t in [0, 1], coefficients low order first, evaluated by Horner and clamped to [0, 1]
(physics.js cpoly6 on the GPU, common.js cmapRGB on the CPU).

  magma  matplotlib's own table (the map the offline |u| movies were made with), fitted by
         least squares at 256 points exactly as common.js's viridis and RdBu were:
             np.polyfit(t, rgb[:, c], 6)[::-1]
         The fit is not the table; its maximum channel error is printed and recorded.
  brat   #8ACE00 -> black, linear in t: rgb = (1 - t) * (138, 206, 0) / 255 (brat_r is its reverse). It is EXACT in this
         form (only the linear coefficient is non-zero), monotone in every channel and so
         in lightness, black at zero.

The fixture holds matplotlib's magma at 256 points, so devtools/checkmusic.js can hold the
coefficients in music.js to the table without matplotlib. Coefficients are rounded to 6
decimals, as the shared table's are; the recorded error is that of the ROUNDED fit.
"""
import json
import os

import numpy as np
import matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
t = np.linspace(0.0, 1.0, 256)
rgb = matplotlib.colormaps["magma"](t)[:, :3]
coef = np.array([np.polyfit(t, rgb[:, c], 6)[::-1] for c in range(3)]).T      # (7, 3), low order first
coef = np.round(coef, 6)
fit = np.clip(sum(coef[k][None, :] * t[:, None] ** k for k in range(7)), 0.0, 1.0)
err = float(np.max(np.abs(fit - rgb)))
green = np.round(np.array([0x8A, 0xCE, 0x00]) / 255.0, 6)
brat = np.zeros((7, 3))                        # the cover: green at 0, black at 1 -- (1 - t) * green
brat[0], brat[1] = green, -green
brat_r = np.zeros((7, 3))                      # reversed: black at 0, green at 1 -- t * green
brat_r[1] = green

fmt = lambda a: "[\n" + ",\n".join("      [" + ", ".join("%.6f" % v for v in row) + "]" for row in a) + "\n    ]"
print("magma: max |fit - table| over 256 points and 3 channels = %.4f" % err)
print("  { name: \"magma\", coef: " + fmt(coef) + " },")
print("  { name: \"brat\", coef: " + fmt(brat) + " },")
print("  { name: \"brat_r\", coef: " + fmt(brat_r) + " }")
dst = os.path.join(HERE, "devtools", "fixtures", "music_cmap.json")
with open(dst, "w") as fh:
    json.dump(dict(source="matplotlib %s colormaps['magma'], 256 points" % matplotlib.__version__,
                   magma_table=[[round(float(v), 6) for v in row] for row in rgb],
                   magma_coef=[[float(v) for v in row] for row in coef], magma_fit_max_error=err,
                   brat_coef=[[float(v) for v in row] for row in brat], brat_hex="#8ACE00",
                   brat_r_coef=[[float(v) for v in row] for row in brat_r]), fh)
print("wrote", dst, os.path.getsize(dst), "bytes")
