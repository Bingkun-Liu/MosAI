#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础版：误差矩阵 + PCA 2D
------------------------
1) 读取各模型 embeddings 与 filenames；读取 .mat 标签（取 stem，对齐，过滤 NaN/Inf）。
2) 在“所有模型 + 有效标签”的交集上做一次 train/test 划分（同一 TEST 用于所有模型）。
3) 各模型用同一 TRAIN 训练，在同一 TEST 预测；收集 TEST 的平方误差 (y_true - y_pred)^2。
4) 把各模型误差拼成 n×m 矩阵（行=测试集图片，列=模型），保存 CSV。
5) 对误差矩阵做 PCA→2D，并保存散点图 PNG 与坐标 CSV。
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


# ===== 基本配置（按需修改） =====
MAT_PATH  = "/Users/kristin/Desktop/vgg_project/image_mean_rating.mat"
OUT_DIR   = "/Users/kristin/Desktop/vgg_new/error_matrix_basic"

MODEL_DIRS: Dict[str, str] = {
    "alexnet":     "/Users/kristin/Desktop/vgg_new/AlexNet-embeddings",
    "git_base":    "/Users/kristin/Desktop/vgg_new/GIT-embeddings",
    "blip2":       "/Users/kristin/Desktop/vgg_new/BLIP2-embeddings",
    "clip":        "/Users/kristin/Desktop/vgg_new/CLIP-B32-embeddings",
    "ResNet50":    "/Users/kristin/Desktop/vgg_new/ResNet50-embeddings",
    "InceptionV3": "/Users/kristin/Desktop/vgg_new/Inception-embeddings",
    "VGG16":       "/Users/kristin/Desktop/vgg_new/VGG16-embeddings",
    "ViT-B16":     "/Users/kristin/Desktop/vgg_new/ViT-B16-embeddings",
}

TEST_SIZE = 0.2
SEED      = 42
# ===========================


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("l2",     Normalizer(norm="l2")),
        ("scaler", StandardScaler()),
        ("ridge",  RidgeCV(alphas=np.logspace(-4, 6, 60)))
    ])


def load_embeddings(model_dir: str) -> Tuple[np.ndarray, List[str]]:
    X = np.load(os.path.join(model_dir, "embeddings.npy")).astype(np.float32)
    with open(os.path.join(model_dir, "filenames.txt"), "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if X.shape[0] != len(names):
        raise ValueError(f"[{model_dir}] embeddings 行数与文件名数不一致: {X.shape[0]} vs {len(names)}")
    return X, names


def load_labels_from_mat(mat_path: str) -> Dict[str, float]:
    mat = sio.loadmat(mat_path)
    stems = [os.path.splitext(str(x[0]).strip())[0] for x in mat["image_names"][0]]
    if "mean_score" in mat:
        scores = mat["mean_score"].flatten().astype(np.float32)
    elif "mean_rating" in mat:
        scores = mat["mean_rating"].flatten().astype(np.float32)
    else:
        raise ValueError("MAT 缺少 mean_score/mean_rating")
    if len(stems) != len(scores):
        raise ValueError("MAT image_names 与 scores 数量不一致")
    m = {}
    for s, v in zip(stems, scores):
        if np.isfinite(v):
            m[s] = float(v)
    return m


def names_to_maps(names: List[str]) -> Tuple[Dict[str, int], Dict[str, str]]:
    stem2idx, stem2name = {}, {}
    for i, fn in enumerate(names):
        st = Path(fn).stem
        if st not in stem2idx:
            stem2idx[st] = i
            stem2name[st] = fn
    return stem2idx, stem2name


def main():
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 标签
    score_map = load_labels_from_mat(MAT_PATH)

    # 2) 读各模型并建立映射
    model_data: Dict[str, Dict] = {}
    for mname, mdir in MODEL_DIRS.items():
        X, names = load_embeddings(mdir)
        stem2idx, stem2name = names_to_maps(names)
        model_data[mname] = dict(X=X, stem2idx=stem2idx, stem2name=stem2name)

    # 3) 共同集合 + 统一划分（一次）
    common = set(score_map.keys())
    for d in model_data.values():
        common &= set(d["stem2idx"].keys())
    stems = sorted(common)
    y_all = np.array([score_map[s] for s in stems], dtype=np.float32)

    idx = np.arange(len(stems))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED)

    stems_te = [stems[i] for i in te_idx]
    y_true_te = y_all[te_idx]

    # 4) 各模型同一 TRAIN/TEST，收集 TEST 误差
    model_order: List[str] = []
    err_cols: List[np.ndarray] = []
    rep_filenames: List[str] = []

    for mname, d in model_data.items():
        X = d["X"]; s2i = d["stem2idx"]; s2n = d["stem2name"]
        idxs = [s2i[s] for s in stems]
        names_ordered = [s2n[s] for s in stems]

        X_all = X[np.array(idxs, dtype=int)]
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        pipe = build_pipeline()
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te).astype(np.float32)

        sq_err = (y_te - y_pred) ** 2
        err_cols.append(sq_err)
        model_order.append(mname)

        if not rep_filenames:
            rep_filenames = [names_ordered[i] for i in te_idx]

    # 5) 误差矩阵 + 保存
    E = np.column_stack(err_cols).astype(np.float32)          # (n_test, m_models)
    err_df = pd.DataFrame({"stem": stems_te,
                           "filename": rep_filenames,
                           "y_true": y_true_te})
    for j, mname in enumerate(model_order):
        err_df[f"err_{mname}"] = E[:, j]

    err_csv = out_dir / "error_matrix_TEST.csv"
    err_df.to_csv(err_csv, index=False)
    print(f"[Saved] {err_csv}  shape={E.shape}")

    # 6) PCA → 2D 可视化 + 坐标
    Z = PCA(n_components=2, random_state=SEED).fit_transform(E)   # (n_test, 2)
    mean_err = E.mean(axis=1)

    coords = pd.DataFrame({
        "x": Z[:, 0], "y": Z[:, 1],
        "mean_err": mean_err,
        "stem": stems_te,
        "filename": rep_filenames
    })
    coords.to_csv(out_dir / "error_PCA_coords.csv", index=False)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=mean_err, s=24)
    plt.colorbar(sc, label="mean squared error across models")
    plt.title("PCA on Error Matrix (TEST)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_dir / "error_PCA_2D.png", dpi=160)
    plt.close()
    print(f"[Saved] {out_dir / 'error_PCA_2D.png'}")


if __name__ == "__main__":
    main()
