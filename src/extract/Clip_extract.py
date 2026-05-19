"""
extract_clip_b32.py
===================
目标：
  使用 open_clip 的 CLIP ViT-B/32 提取每张图片的最终图像向量，
  保存到 embeddings.npy (N, D) 与 filenames.txt (N)。

一致性：
  - 位点：视觉编码器 encode_image 的输出（全局图像表征）
  - 不做 L2/标准化；回归阶段统一处理
"""

import os
import numpy as np
from PIL import Image
import torch
import open_clip

# ===== 1) 写死路径与批大小 =====
IMAGE_FOLDER  = "/Users/kristin/Desktop/vgg_project/Prady_art_dataset"
OUTPUT_FOLDER = "/Users/kristin/Desktop/vgg_new/CLIP-B32-embeddings"
BATCH_SIZE    = 16
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===== 2) 设备选择 =====
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using {device}")

# ===== 3) 加载 CLIP 模型与预处理 =====
# 你也可以换 ViT-L/14：("ViT-L-14", pretrained="openai")
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
model.eval()

# ===== 4) 主提取逻辑 =====
@torch.inference_mode()
def extract_embeddings():
    files = [f for f in sorted(os.listdir(IMAGE_FOLDER))
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"))]
    if not files:
        raise RuntimeError(f"未在目录中发现图片：{IMAGE_FOLDER}")

    vecs, names = [], []
    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i:i+BATCH_SIZE]

        # 读图 + 预处理（open_clip 自带：Resize/Crop/Normalize）
        xs = [preprocess(Image.open(os.path.join(IMAGE_FOLDER, f)).convert("RGB")).unsqueeze(0)
              for f in batch]
        x  = torch.cat(xs, dim=0).to(device)   # [B,3,H,W]，通常 H=W=224

        # 视觉编码器 → 最终图像向量  [B, D]（B32 的 D=512）
        z  = model.encode_image(x)
        z  = z.float()                         # 转 float32 以便保存（有些权重默认 fp16）

        f = z.cpu().numpy().astype(np.float32)
        vecs.append(f); names.extend(batch)
        print(f"[Batch {i//BATCH_SIZE+1}] Processed {len(batch)} images")

    X = np.vstack(vecs)  # (N, 512)
    return X, names

# ===== 5) 保存输出 =====
if __name__ == "__main__":
    X, names = extract_embeddings()
    np.save(os.path.join(OUTPUT_FOLDER, "embeddings.npy"), X)
    with open(os.path.join(OUTPUT_FOLDER, "filenames.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names))
    print(f"[Done][CLIP ViT-B/32] 提取完成: X={X.shape}, 已保存到 {OUTPUT_FOLDER}")
