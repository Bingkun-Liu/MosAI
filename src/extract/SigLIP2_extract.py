#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Minimal SigLIP2 extractor (与 BLIP/CLIP 简洁风格一致)
# 依赖: pip install transformers pillow torch numpy

import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

# ---- 路径配置（按需修改）----
IMAGE_FOLDER  = "/Users/kristin/Desktop/vgg_project/Prady_art_dataset"
OUTPUT_FOLDER = "/Users/kristin/Desktop/vgg_new/SigLIP2-embeddings"

# 常用 SigLIP2 权重（384 分辨率）
MODEL_ID   = "google/siglip-so400m-patch14-384"
BATCH_SIZE = 16
EXTS       = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")

def list_images(root):
    p = Path(root)
    files = []
    for ext in EXTS:
        files += [str(x) for x in p.rglob(f"*{ext}")]
    # 与其他模型对齐：按文件名排序
    return sorted(files, key=lambda s: Path(s).name.lower())

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    paths = list_images(IMAGE_FOLDER)
    if not paths:
        raise FileNotFoundError(f"在 {IMAGE_FOLDER} 未找到图片")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model     = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    all_vecs = []
    all_names = []

    with torch.inference_mode():
        for i in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[i:i+BATCH_SIZE]
            images = [Image.open(p).convert("RGB") for p in batch_paths]

            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            # 优先用 get_image_features（若无则回退）
            if hasattr(model, "get_image_features"):
                feats = model.get_image_features(pixel_values=pixel_values)   # [B, D]
            else:
                out = model(pixel_values=pixel_values)
                feats = out.image_embeds if hasattr(out, "image_embeds") else out[0]

            all_vecs.append(feats.cpu().float().numpy())
            all_names += [Path(p).name for p in batch_paths]

            print(f"[Batch {i//BATCH_SIZE + 1}] Processed {len(batch_paths)} images")

    X = np.vstack(all_vecs).astype(np.float32)
    np.save(os.path.join(OUTPUT_FOLDER, "embeddings.npy"), X)
    with open(os.path.join(OUTPUT_FOLDER, "filenames.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_names))
    print(f"[Done][SigLIP2] 提取完成: X={X.shape}, 已保存到 {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
