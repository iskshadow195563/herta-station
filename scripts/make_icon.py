# -*- coding: utf-8 -*-
"""Draw the Herta Station hex-sigil icon and export a multi-size .ico.

    python scripts/make_icon.py     ->  assets/herta.ico
"""

import math
import os

from PIL import Image, ImageDraw, ImageFilter

SIZE = 512
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "herta.ico")

img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

cx = cy = SIZE / 2
R = 220


def hexagon(radius, rot=0.0):
    pts = []
    for i in range(6):
        a = rot + i * math.pi / 3
        pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return pts


d.rounded_rectangle([18, 18, SIZE - 18, SIZE - 18], radius=96, fill=(8, 12, 26, 255))

glow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
gd.polygon(hexagon(R), outline=(111, 242, 255, 235), width=10)
gd.polygon(hexagon(R - 34), outline=(167, 123, 255, 200), width=6)
img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(12)))

d = ImageDraw.Draw(img)
d.polygon(hexagon(R), outline=(150, 248, 255, 255), width=9)
d.polygon(hexagon(R - 36), outline=(167, 123, 255, 235), width=6)

core = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
cd = ImageDraw.Draw(core)
cd.polygon(hexagon(104), fill=(111, 242, 255, 60), outline=(111, 242, 255, 255), width=8)
img.alpha_composite(core.filter(ImageFilter.GaussianBlur(6)))

d = ImageDraw.Draw(img)
d.polygon(hexagon(104), fill=(24, 60, 92, 210), outline=(190, 252, 255, 255), width=7)
d.line([(cx, cy - 62), (cx, cy + 62)], fill=(111, 242, 255, 210), width=6)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
img.save(OUT, sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
print("icon written:", OUT)
