#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 最简 Florence-2 提取脚本：读图 -> AutoProcessor -> get_image_features/vision_model -> 保存
# 依赖：pip install transformers pillow numpy torch

import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel

# --- 路径与配置（按需改） ---
IMAGE_FOLDER  = "/Users/kristin/Desktop/vgg_project/Prady_art_dataset"
OUTPUT_FOLDER = "/Users/kristin/Desktop/vgg_new/Florence2-embeddings"

MODEL_ID   = "florence-community/Florence-2-base"  # 社区镜像，兼容性更好
BATCH_SIZE = 16

# --- 设备 ---
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1) 列图 & 排序
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
    paths = sorted([str(p) for p in Path(IMAGE_FOLDER).rglob("*") if p.suffix.lower() in exts],
                   key=lambda s: Path(s).name.lower())
    if not paths:
        raise FileNotFoundError(f"No images found in {IMAGE_FOLDER}")

    # 2) 处理器 + 模型
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE).eval()

    feats_all = []
    names_all = []

    # 3) 批处理（超简）
    with torch.no_grad():
        for i in range(0, len(paths), BATCH_SIZE):
            batch = paths[i:i+BATCH_SIZE]
            imgs = [Image.open(p).convert("RGB") for p in batch]

            # Florence-2 官方预处理：显式取 pixel_values，避免之前的“布尔张量歧义”
            px = processor(images=imgs, return_tensors="pt")
            if "pixel_values" in px:
                pixel_values = px["pixel_values"].to(DEVICE)
            elif "images" in px:
                pixel_values = px["images"].to(DEVICE)
            else:
                raise RuntimeError(f"Processor output missing image tensor. Keys: {list(px.keys())}")

            # 4) 拿全局图像向量：优先 get_image_features；否则走 vision_model + 池化
            if hasattr(model, "get_image_features"):
                emb = model.get_image_features(pixel_values=pixel_values)  # [B, D] 或 [B, T, D]
            else:
                out = model.vision_model(pixel_values=pixel_values)
                emb = getattr(out, "pooler_output", None)
                if emb is None:
                    emb = out.last_hidden_state  # [B, T, D]

            # 若是序列特征 -> mean-pool 到 [B, D]
            if emb.dim() == 3:
                emb = emb.mean(dim=1)

            feats_all.append(emb.float().cpu().numpy().astype(np.float32))
            names_all.extend([Path(p).name for p in batch])

            print(f"[Batch {i//BATCH_SIZE + 1}] {len(batch)} images")

    X = np.vstack(feats_all).astype(np.float32)  # [N, D]

    # 5) 保存
    np.save(os.path.join(OUTPUT_FOLDER, "embeddings.npy"), X)
    with open(os.path.join(OUTPUT_FOLDER, "filenames.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names_all))

    print(f"[Done][Florence-2] X={X.shape}, saved to {OUTPUT_FOLDER}")
    print("emb shape:", emb.shape)
    print("has get_image_features:", hasattr(model, "get_image_features"))
    print("keys:", list(px.keys()))
    print("has pooler:", hasattr(model.vision_model, "post_layernorm") or hasattr(out, "pooler_output"))
