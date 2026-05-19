#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_imagebind.py
====================
目标：
  与 BLIP-2 提取脚本保持一致的接口/输出，仅使用 ImageBind 的视觉分支提取图像向量，
  保存到 embeddings.npy (N, D) 与 filenames.txt (N)。

一致性：
  - 位点：ImageBind 视觉嵌入（model(inputs)[ModalityType.VISION]）
  - 不做 L2/标准化；回归阶段统一处理
  - 批量处理 + 文件名排序 + 跳过坏图

依赖：
  pip install torch pillow numpy
  pip install git+https://github.com/facebookresearch/ImageBind
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch

# ---- ImageBind ----
try:
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
    from imagebind import data as imagebind_data
except Exception as e:
    raise RuntimeError(
        "未找到 ImageBind 依赖。请先安装：\n"
        "  pip install git+https://github.com/facebookresearch/ImageBind\n"
        f"原始错误：{e}"
    )

# ============ 配置（与你现有脚本风格一致） ============
IMAGE_FOLDER   = "/Users/kristin/Desktop/vgg_project/Prady_art_dataset"               # 改成你的图片目录
OUTPUT_FOLDER  = "/Users/kristin/Desktop/vgg_new/ImageBind-embeddings"      # 建议每个模型单独目录
BATCH_SIZE     = 16
#SEED           = 42
# 设备与精度
def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _get_device()
DTYPE  = torch.float32  # ImageBind 官方实现以 float32 为主，保持稳妥

# 允许的图片扩展名（与你评测脚本一致）
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ============ 工具函数 ============
def _list_images(img_dir: str) -> List[str]:
    p = Path(img_dir)
    files = []
    for ext in EXTS:
        files.extend([str(x) for x in p.rglob(f"*{ext}")])
    # 按文件名排序（可复现且与其他模型对齐）
    files = sorted(files, key=lambda s: Path(s).name.lower())
    return files

def _ensure_outdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

# ============ 核心：提取函数 ============
def extract_embeddings() -> Tuple[np.ndarray, List[str]]:
    #torch.manual_seed(SEED)

    img_paths = _list_images(IMAGE_FOLDER)
    if not img_paths:
        raise FileNotFoundError(f"在 {IMAGE_FOLDER} 未找到有效图片（扩展名：{sorted(EXTS)}）")
    _ensure_outdir(OUTPUT_FOLDER)

    # 1) 加载 ImageBind 模型（huge 版），只用视觉分支
    model = imagebind_model.imagebind_huge(pretrained=True)
    model = model.to(DEVICE)
    model.eval()

    all_vecs: List[np.ndarray] = []
    all_names: List[str] = []

    # 2) 小工具：逐批安全载入（坏图跳过）
    def _safe_batch(batch_paths: List[str]) -> List[str]:
        ok = []
        for p in batch_paths:
            try:
                with Image.open(p) as im:
                    im.verify()  # 只验证，不真正解码
                ok.append(p)
            except Exception as e:
                print(f"[WARN] 跳过坏图: {p} ({e})")
        return ok

    # 3) 按批处理
    N = len(img_paths)
    for i in range(0, N, BATCH_SIZE):
        batch_paths = img_paths[i:i + BATCH_SIZE]
        batch_paths = _safe_batch(batch_paths)
        if not batch_paths:
            continue

        # ImageBind 自带的预处理（会返回已张量化的 pixel_values）
        inputs = {
            ModalityType.VISION: imagebind_data.load_and_transform_vision_data(   # Resize(短边=224) → CenterCrop(224×224) → ToTensor → Normalize(CLIP mean/std)
                batch_paths, device=DEVICE
            ).to(dtype=DTYPE)
        }

        with torch.inference_mode():
            out = model(inputs)  # dict: modality -> Tensor
            emb = out[ModalityType.VISION]  # [B, D]，已是全局向量
            # 与你的管线一致：不做 L2/标准化，这里只转 float32 + numpy
            f = emb.detach().to("cpu").to(torch.float32).numpy().astype(np.float32)

        all_vecs.append(f)
        all_names.extend([Path(p).name for p in batch_paths])

        print(f"[Batch {i//BATCH_SIZE + 1}] Processed {len(batch_paths)} images")

    if not all_vecs:
        raise RuntimeError("没有成功处理任何图片，请检查图片目录或依赖安装。")

    X = np.vstack(all_vecs).astype(np.float32)  # (N_eff, D)
    return X, all_names

# ============ 主程序 ============
if __name__ == "__main__":
    X, names = extract_embeddings()
    np.save(os.path.join(OUTPUT_FOLDER, "embeddings.npy"), X)
    with open(os.path.join(OUTPUT_FOLDER, "filenames.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names))
    print(f"[Done][ImageBind] 提取完成: X={X.shape}, 已保存到 {OUTPUT_FOLDER}")
