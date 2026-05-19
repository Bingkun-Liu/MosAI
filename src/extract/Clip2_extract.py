#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 最简 OpenCLIP 提取：读图 -> 官方预处理 -> encode_image -> 保存
# 依赖：pip install open_clip_torch pillow numpy torch

import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import open_clip

# --- 路径与配置（按需改） ---
IMAGE_FOLDER  = "/Users/kristin/Desktop/vgg_project/Prady_art_dataset"
OUTPUT_FOLDER = "/Users/kristin/Desktop/vgg_new/CLIP-B16-embeddings"

MODEL_NAME  = "ViT-B-16"            # 也可: "ViT-B-32", "ViT-L-14", ...
PRETRAINED  = "laion2b_s34b_b88k"   # 也可: "openai", "laion2b_s32b_b82k", "datacomp_xl_s13b_b90k"
BATCH_SIZE  = 16

# --- 设备 ---
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# --- 主流程（超简） ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1) 列图 & 排序
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
    paths = sorted([str(p) for p in Path(IMAGE_FOLDER).rglob("*") if p.suffix.lower() in exts],
                   key=lambda s: Path(s).name.lower())
    if not paths:
        raise FileNotFoundError(f"No images found in {IMAGE_FOLDER}")

    # 2) 模型 & 官方预处理（val变换）
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(DEVICE).eval()

    all_feats = []
    all_names = []

    # 3) 批处理
    with torch.no_grad():
        for i in range(0, len(paths), BATCH_SIZE):
            batch = paths[i:i+BATCH_SIZE]

            # 读图 + 预处理（和官方一致）
            imgs = [preprocess(Image.open(p).convert("RGB")) for p in batch]
            pixel_values = torch.stack(imgs, dim=0).to(DEVICE)

            # 前向：图像全局向量
            feats = model.encode_image(pixel_values)               # [B, D]
            feats = feats.float().cpu().numpy().astype(np.float32) # 不做L2/标准化

            all_feats.append(feats)
            all_names.extend([Path(p).name for p in batch])

            print(f"[Batch {i//BATCH_SIZE + 1}] {len(batch)} images")

    X = np.vstack(all_feats).astype(np.float32)  # [N, D]

    # 4) 保存
    np.save(os.path.join(OUTPUT_FOLDER, "embeddings.npy"), X)
    with open(os.path.join(OUTPUT_FOLDER, "filenames.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_names))

    print(f"[Done][OpenCLIP] X={X.shape}, saved to {OUTPUT_FOLDER}")
