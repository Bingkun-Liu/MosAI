# -*- coding: utf-8 -*-
"""
Scatter of Pearson correlations grouped by pair type (excl. GIT & Florence2):
- L-L: vision-language vs vision-language
- V-L: vision-only vs vision-language
- V-V: vision-only vs vision-only

No seaborn; single matplotlib figure; no explicit colors.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product

# ---------- 1) 定义模型分组 ----------
# L: BLIP, CLIP, SigLIP2, ImageBind, Clip2
# V: ViT-B/16, Inception v3, AlexNet, ResNet50, VGG
# ---------- 1) 定义模型分组（新数据） ----------
# L: 视-语模型
L = ["BLIP2", "clip", "Clip2", "SigLIP2", "ImageBind"]  # "Florence2", "git_base"
# V: 纯视觉模型
V = ["ViT-B16", "InceptionV3", "alexnet", "ResNet50", "VGG16"]

# ---------- 2) 相关系数表（新数据） ----------
C = {
    # Florence2 与其它（如需纳入可用；当前分组里排除了 Florence2）
    ("Florence2","AlexNet"):0.965543, ("Florence2","BLIP2"):0.966728, ("Florence2","clip"):0.967343,
    ("Florence2","Clip2"):0.96691,  ("Florence2","git_base"):0.971465, ("Florence2","ImageBind"):0.962851,
    ("Florence2","InceptionV3"):0.963292, ("Florence2","ResNet50"):0.963861, ("Florence2","SigLIP2"):0.965763,
    ("Florence2","VGG16"):0.963792, ("Florence2","ViT-B16"):0.962906,

    # alexnet 与其它
    ("alexnet","BLIP2"):0.994131, ("alexnet","clip"):0.99671,  ("alexnet","Clip2"):0.996345,
    ("alexnet","git_base"):0.973682, ("alexnet","ImageBind"):0.987413, ("alexnet","InceptionV3"):0.996035,
    ("alexnet","ResNet50"):0.997712, ("alexnet","SigLIP2"):0.996611, ("alexnet","VGG16"):0.99846,
    ("alexnet","ViT-B16"):0.995812,

    # BLIP2 与其它
    ("BLIP2","clip"):0.997535, ("BLIP2","Clip2"):0.998022, ("BLIP2","git_base"):0.972309,
    ("BLIP2","ImageBind"):0.9925, ("BLIP2","InceptionV3"):0.993483, ("BLIP2","ResNet50"):0.995446,
    ("BLIP2","SigLIP2"):0.997733, ("BLIP2","VGG16"):0.995593, ("BLIP2","ViT-B16"):0.993234,

    # clip 与其它
    ("clip","Clip2"):0.999153, ("clip","git_base"):0.972845, ("clip","ImageBind"):0.991495,
    ("clip","InceptionV3"):0.996713, ("clip","ResNet50"):0.997073, ("clip","SigLIP2"):0.998801,
    ("clip","VGG16"):0.997749, ("clip","ViT-B16"):0.996471,

    # Clip2 与其它
    ("Clip2","git_base"):0.973342, ("Clip2","ImageBind"):0.992916, ("Clip2","InceptionV3"):0.996198,
    ("Clip2","ResNet50"):0.996596, ("Clip2","SigLIP2"):0.999005, ("Clip2","VGG16"):0.997435,
    ("Clip2","ViT-B16"):0.995978,

    # git_base 与其它（如需纳入可用；当前分组里排除了 git_base）
    ("git_base","ImageBind"):0.966513, ("git_base","InceptionV3"):0.975083, ("git_base","ResNet50"):0.973511,
    ("git_base","SigLIP2"):0.973246, ("git_base","VGG16"):0.974929, ("git_base","ViT-B16"):0.975032,

    # ImageBind 与其它
    ("ImageBind","InceptionV3"):0.983626, ("ImageBind","ResNet50"):0.988945, ("ImageBind","SigLIP2"):0.990678,
    ("ImageBind","VGG16"):0.987686, ("ImageBind","ViT-B16"):0.983212,

    # InceptionV3 与其它
    ("InceptionV3","ResNet50"):0.996641, ("InceptionV3","SigLIP2"):0.997869, ("InceptionV3","VGG16"):0.998082,
    ("InceptionV3","ViT-B16"):0.999978,

    # ResNet50 与其它
    ("ResNet50","SigLIP2"):0.997552, ("ResNet50","VGG16"):0.998896, ("ResNet50","ViT-B16"):0.996411,

    # SigLIP2 与其它
    ("SigLIP2","VGG16"):0.998262, ("SigLIP2","ViT-B16"):0.997725,

    # VGG16 与 ViT-B16
    ("VGG16","ViT-B16"):0.997932,
}


def corr(a, b):
    if (a, b) in C: return C[(a, b)]
    if (b, a) in C: return C[(b, a)]
    raise KeyError(f"Missing correlation for pair ({a}, {b})")

# ---------- 3) 拆成三类 ----------
LL = [corr(a, b) for (a, b) in combinations(L, 2)]
VL = [corr(a, b) for (a, b) in product(L, V)]
VV = [corr(a, b) for (a, b) in combinations(V, 2)]

groups   = [VL, LL, VV]
x_labels = ["V–L", "L–L", "V–V"]

# ---------- 4) 画图（竖向散点 + 组均值线） ----------
fig, ax = plt.subplots(figsize=(6.2, 5.0))

rng = np.random.default_rng(42)
x_pos = np.arange(len(groups))          # 0,1,2
markers = ["o", "s", "D"]               # 每组不同形状

for i, vals in enumerate(groups):
    vals = np.asarray(vals, dtype=float)
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    xx = np.full_like(vals, x_pos[i], dtype=float) + jitter
    ax.scatter(xx, vals, marker=markers[i], label=x_labels[i])
    m = float(np.mean(vals))
    ax.hlines(m, x_pos[i]-0.18, x_pos[i]+0.18, linewidth=2)

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)
ax.set_xlim(-0.5, len(groups)-0.5)
ax.set_ylim(0.0, 1.0)
ax.set_ylabel("Pearson correlation")

ax.text(1.5, 0.98,
        "L: BLIP / CLIP / SigLIP2 / ImageBind / Clip2\nV: ViT-B/16 / Inception v3 / AlexNet / ResNet50 / VGG",
        va="top", ha="left")

for spine in ["left", "bottom"]:
    ax.spines[spine].set_linewidth(2.0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("pearson_group_scatter_excl_GIT_F2.png", dpi=300)
plt.show()
print("Saved: pearson_group_scatter_excl_GIT_F2.png")
