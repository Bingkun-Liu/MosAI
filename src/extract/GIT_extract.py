"""
GIT_extract.py
==============
目标：
  从指定文件夹读取图片，只使用 Microsoft GIT 的“视觉编码器”（GitVisionModel）
  提取每张图片的全局图像向量（CLS 向量），并保存为：
    - embeddings.npy : (N, D)  的 numpy 矩阵（git-base 的 D=768）
    - filenames.txt  : N 行文件名，顺序与 embeddings 一一对应

统一性（和你其它模型保持一致）：
  - 位点：Transformer 系统一用 “最终层 CLS 向量”（若无 pooler 则从 last_hidden_state 取第 0 个 token）
  - 预处理：使用 checkpoint 自带的 AutoImageProcessor（保证与训练分布一致）
  - 这里不做 L2/StandardScaler；这些在回归脚本里统一做（跨模型可比）
"""

import os
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, GitVisionModel

# ======================
# 1) 写死路径 & 批大小
# ======================
IMAGE_FOLDER  = "/Users/kristin/Desktop/vgg_project/Prady_art_dataset"   # 输入图片目录
OUTPUT_FOLDER = "/Users/kristin/Desktop/vgg_new/GIT-embeddings"          # 输出目录
BATCH_SIZE    = 16                                                       # 每批处理的图片张数
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ======================
# 2) 选择设备与精度
# ======================
# - CUDA 可用时用 GPU，并启用 float16（更省显存/更快）
# - MPS/CPU 上为稳妥统一用 float32（某些算子对 fp16 支持不佳）
use_fp16 = torch.cuda.is_available()
dtype    = torch.float16 if use_fp16 else torch.float32
device   = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[Device] Using {device}, dtype={dtype}")

# ======================
# 3) 加载“视觉编码器”与官方预处理器
# ======================
# 只加载视觉编码器（GitVisionModel），不加载整套多模态模型（GitModel）
# 好处：显存占用更小，且我们只需要视觉向量做回归
CKPT = "microsoft/git-base"  # 可改 "microsoft/git-large"（向量维度会变，内存更大）

# AutoImageProcessor 会根据 ckpt 配置自动做 resize/crop/normalize 到期望输入尺寸（git-base 通常 224×224）
processor = AutoImageProcessor.from_pretrained(CKPT)

# 视觉编码器（本质是一个 ViT 视觉塔），前向输出包含 last_hidden_state（[B, T, C]）
vision = GitVisionModel.from_pretrained(CKPT, torch_dtype=dtype, low_cpu_mem_usage=True)
vision.eval().to(device)

# ======================
# 4) 主提取函数：读图 -> 预处理 -> 前向 -> 取 CLS -> 累积
# ======================
@torch.inference_mode()  # 关闭梯度，节省内存并加速
def extract_embeddings():
    """
    遍历 IMAGE_FOLDER，分批提取 GIT 视觉向量：
      返回：
        X: numpy.ndarray, 形状 (N, D)  其中 D 对 git-base 为 768
        names: list[str], 长度 N，文件名顺序与 X 的行一一对应
    """
    # 只收集常见图片格式（按字母序排序，保证可复现）
    files = [f for f in sorted(os.listdir(IMAGE_FOLDER))
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"))]
    if not files:
        raise RuntimeError(f"未在目录中发现图片：{IMAGE_FOLDER}")

    vecs, names = [], []

    for i in range(0, len(files), BATCH_SIZE):
        batch_files = files[i:i + BATCH_SIZE]

        # (1) 读图并用官方处理器转为 pixel_values（[B,3,H,W]，git-base 默认 224×224）
        imgs = [Image.open(os.path.join(IMAGE_FOLDER, fname)).convert("RGB")
                for fname in batch_files]
        inputs = processor(images=imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)  # [B,3,H,W]

        # (2) 仅过视觉塔，得到最后一层隐藏状态 [B, T, C]
        #     T = token 数（patch 数 + 1 个 CLS）
        vout   = vision(pixel_values=pixel_values)
        hidden = vout.last_hidden_state                           # [B, T, C]

        # (3) 取 CLS 向量：第 0 个 token → [B, C]
        #     这就是整张图的全局语义表示，和 ViT/DINOv2/BLIP-2 的 CLS 位点一致
        cls = hidden[:, 0, :]                                     # [B, C]（git-base: C=768）

        # (4) 转为 numpy.float32 并累积
        f = cls.detach().float().cpu().numpy().astype(np.float32) # [B, C]
        vecs.append(f)
        names.extend(batch_files)

        print(f"[Batch {i // BATCH_SIZE + 1}] Processed {len(batch_files)} images")

    # (5) 拼接所有批次 → (N, C)
    X = np.vstack(vecs)
    return X, names

# ======================
# 5) 运行 & 保存
# ======================
if __name__ == "__main__":
    X, names = extract_embeddings()  # X: (N, 768) for git-base

    # 保存向量矩阵
    np.save(os.path.join(OUTPUT_FOLDER, "embeddings.npy"), X)

    # 保存文件名（行顺序与 X 对齐，供回归阶段按名字对齐标签）
    with open(os.path.join(OUTPUT_FOLDER, "filenames.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names))

    print(f"[Done][GIT-Base] 提取完成: X={X.shape}, 已保存到 {OUTPUT_FOLDER}")
