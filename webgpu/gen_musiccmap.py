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
  brat   black -> #8ACE00, linear in t: rgb = t * (138, 206, 0) / 255. It is EXACT in this
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
brat = np.zeros((7, 3))
brat[1] = np.round(np.array([0x8A, 0xCE, 0x00]) / 255.0, 6)

fmt = lambda a: "[\n" + ",\n".join("      [" + ", ".join("%.6f" % v for v in row) + "]" for row in a) + "\n    ]"
print("magma: max |fit - table| over 256 points and 3 channels = %.4f" % err)
print("  { name: \"magma\", coef: " + fmt(coef) + " },")
print("  { name: \"brat\", coef: " + fmt(brat) + " }")
dst = os.path.join(HERE, "devtools", "fixtures", "music_cmap.json")
with open(dst, "w") as fh:
    json.dump(dict(source="matplotlib %s colormaps['magma'], 256 points" % matplotlib.__version__,
                   magma_table=[[round(float(v), 6) for v in row] for row in rgb],
                   magma_coef=[[float(v) for v in row] for row in coef], magma_fit_max_error=err,
                   brat_coef=[[float(v) for v in row] for row in brat], brat_hex="#8ACE00"), fh)
print("wrote", dst, os.path.getsize(dst), "bytes")
