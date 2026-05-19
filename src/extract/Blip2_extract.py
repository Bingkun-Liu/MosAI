"""
extract_blip2.py
================
目标：
  对每张图片，仅通过 BLIP-2 的视觉塔（CLIPVisionModel）提取 pooled_output（CLS 池化向量），
  保存到 embeddings.npy (N, D) 与 filenames.txt (N)。

一致性：
  - 位点：视觉塔 pooled_output（等价于 CLS 池化）
  - 不做 L2/标准化；回归阶段统一处理
"""

import os
import numpy as np
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2Model

# ===== 1) 写死路径与批大小 =====
IMAGE_FOLDER  = "/Users/kristin/Desktop/vgg_project/Prady_art_dataset"
OUTPUT_FOLDER = "/Users/kristin/Desktop/vgg_new/BLIP2-embeddings"
BATCH_SIZE    = 8   # BLIP-2 视觉塔分辨率较高，建议 batch 小一些
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===== 2) 设备与精度 =====
# fp16 在 MPS 上部分算子不支持时可能报错；报错就改回 float32
use_fp16 = torch.cuda.is_available()          # 只有 CUDA 时默认 fp16
dtype    = torch.float16 if use_fp16 else torch.float32
device   = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[Device] Using {device}, dtype={dtype}")

# ===== 3) 选择 BLIP-2 checkpoint =====
# 这些权重都可以；越大的语言模型越重，但我们只用 vision_model：
#   "Salesforce/blip2-opt-2.7b"
#   "Salesforce/blip2-flan-t5-xl"
#   "Salesforce/blip2-flan-t5-xxl"
CKPT = "Salesforce/blip2-opt-2.7b"

# Processor 负责图像预处理（resize/crop/normalize 到 vision 塔期望的输入尺寸）
processor = Blip2Processor.from_pretrained(CKPT)

# 仅作“容器”，我们只用里面的 vision_model；low_cpu_mem_usage=True 降低内存峰值
model = Blip2Model.from_pretrained(CKPT, torch_dtype=dtype, low_cpu_mem_usage=True)
model.eval().to(device)

@torch.inference_mode()
def extract_embeddings():
    files = [f for f in sorted(os.listdir(IMAGE_FOLDER))
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"))]
    if not files:
        raise RuntimeError(f"未在目录中发现图片：{IMAGE_FOLDER}")

    vecs, names = [], []

    # 方便拿到视觉塔
    vision = model.vision_model  # 这是 CLIPVisionModel（或同类视觉 backbone）

    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i:i+BATCH_SIZE]

        # 1) 预处理到 pixel_values（Processor 会按 checkpoint 的规范做 resize/crop/normalize）
        imgs = [Image.open(os.path.join(IMAGE_FOLDER, f)).convert("RGB") for f in batch]
        inputs = processor(images=imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)  # [B, 3, H, W]

        # 2) 只过视觉塔（不走 Q-Former / 语言模型）：拿 pooled_output（CLS 池化）
        vout = vision(pixel_values=pixel_values)
        # 常见输出：BaseModelOutputWithPooling(last_hidden_state, pooler_output, ...)
        if hasattr(vout, "pooler_output") and vout.pooler_output is not None:
            emb = vout.pooler_output             # [B, D]，例如 ViT-g 的 D 可能为 1024/1280 视具体塔
        else:
            # 部分视觉塔可能没有 pooler；退化为 tokens 均值
            emb = vout.last_hidden_state.mean(dim=1)  # [B, D]

        f = emb.detach().float().cpu().numpy().astype(np.float32)
        vecs.append(f); names.extend(batch)

        print(f"[Batch {i//BATCH_SIZE+1}] Processed {len(batch)} images")

    X = np.vstack(vecs)  # (N, D)
    return X, names

if __name__ == "__main__":
    X, names = extract_embeddings()
    np.save(os.path.join(OUTPUT_FOLDER, "embeddings.npy"), X)
    with open(os.path.join(OUTPUT_FOLDER, "filenames.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names))
    print(f"[Done][BLIP-2] 提取完成: X={X.shape}, 已保存到 {OUTPUT_FOLDER}")
