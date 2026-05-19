# -*- coding: utf-8 -*-
"""
画“模型两两 Pearson 相关性”的热力图（不含 GIT）
- 只用 numpy/pandas/matplotlib（无 seaborn）
- 单图（无子图），默认配色，不手动指定颜色
- 每个格子叠加数值标注（两位小数）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 1) 准备数据（把你表格里“去掉 GIT”的数值写成上三角；脚本会补全对称&对角线=1） =========
labels = ["BLIP", "CLIP", "ViT-B/16", "Inception v3", "AlexNet",
    "ResNet50", "VGG", "SigLIP2", "ImageBind", "Clip2"]

# 上三角（含对角占位 None）：行依次对应 labels，列 j>=i
# 用你截图里的数值（对称）；对角线先放 None，下面会自动填 1.0
upper = [
    # BLIP
    [None, 0.623558, 0.523213, 0.587948, 0.431419, 0.567744, 0.456066, 0.660337, 0.771678, 0.585237],
    # CLIP
    [None, None,    0.676647, 0.541073, 0.567152, 0.548473, 0.571092, 0.667819, 0.697482, 0.704596],
    # ViT-B/16
    [None, None,    None,     0.685729, 0.578366, 0.642804, 0.636187, 0.506697, 0.586311, 0.509501],
    # Inception v3
    [None, None,    None,     None,     0.629851, 0.838785, 0.694887, 0.406499, 0.507186, 0.528832],
    # AlexNet
    [None, None,    None,     None,     None,     0.666437, 0.802947, 0.497869, 0.455556, 0.49685],
    # ResNet50
    [None, None,    None,     None,     None,     None,     0.707295, 0.420081, 0.535918, 0.410602],
    # VGG
    [None, None,    None,     None,     None,     None,     None,     0.511058, 0.466762, 0.478537],
    # SigLIP2
    [None, None,    None,     None,     None,     None,     None,     None,     0.727817, 0.659238],
    # ImageBind
    [None, None,    None,     None,     None,     None,     None,     None,     None,     0.685012],
    # Clip2
    [None, None,    None,     None,     None,     None,     None,     None,     None,     None],
]

n = len(labels)
M = np.zeros((n, n), dtype=float)

# 补全矩阵：对角线=1，上三角填数，下三角=对称
for i in range(n):
    for j in range(n):
        if i == j:
            M[i, j] = 1.0
        elif j > i:
            M[i, j] = float(upper[i][j])  # 用上三角
        else:  # j < i
            M[i, j] = M[j, i]            # 对称

df = pd.DataFrame(M, index=labels, columns=labels)
print("相关性矩阵：")
print(df)

# ========= 2) 画热力图 =========
fig, ax = plt.subplots(figsize=(7.5, 6.5))  # 可按需调整尺寸

# imshow 默认配色即可（不手动指定 cmap 颜色）
im = ax.imshow(df.values, aspect='equal')

# 坐标轴刻度与标签
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)

# 网格线（可选）
ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
ax.grid(which='minor', color='w', linewidth=1)
ax.tick_params(which='both', bottom=False, left=False)

# 数值标注（两位小数）
vals = df.values
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{vals[i, j]:.2f}", va='center', ha='center', fontsize=9, color='black')

# 颜色条
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson correlation", rotation=90)

plt.title("Model-wise Pearson Correlation (excl. GIT)")
plt.tight_layout()

# 保存&显示
plt.savefig("model_correlation_heatmap_excl_GIT.png", dpi=300)
plt.show()
print("已保存为 model_correlation_heatmap_excl_GIT.png")
