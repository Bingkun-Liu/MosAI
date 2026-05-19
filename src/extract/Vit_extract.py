"""
extract_vit_b16.py
==================
目标：
  从一个图片文件夹中，使用 timm 的 ViT-B/16（ImageNet 预训练）提取每张图片的“最终层 CLS 向量”，
  并保存为 embeddings.npy (N, D) 和 filenames.txt (N)。

一致性：
  - ViT/DINOv2 类：统一取“最终层 CLS token”的向量（若模型无 CLS，就对所有 tokens 做平均）
  - 不做 L2 / 标准化，留到回归阶段统一处理（保证跨模型可比）
"""

import os
import numpy as np
from PIL import Image
import torch
import timm

# ======================
# 1) 路径和批大小（写死）
# ======================
IMAGE_FOLDER  = "/Users/kristin/Desktop/vgg_project/Prady_art_dataset"
OUTPUT_FOLDER = "/Users/kristin/Desktop/vgg_new/ViT-B16-embeddings"
BATCH_SIZE    = 16
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ======================
# 2) 选择设备
# ======================
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using {device}")

# ======================
# 3) 创建 ViT-B/16 模型 + 官方预处理
# ======================
# timm 模型名：vit_base_patch16_224
# - 输入分辨率：224x224
# - timm.create_model(..., pretrained=True) 会自动加载 ImageNet 预训练权重
model_name = "vit_base_patch16_224"
net = timm.create_model(model_name, pretrained=True)
net.eval().to(device)

# timm 官方的 transforms：把输入 resize/crop 到 224、ToTensor、Normalize 到 ImageNet 均值方差
# 注意：is_training=False 使用评估期变换
preprocess = timm.data.transforms_factory.create_transform(
    input_size=(3, 224, 224),
    is_training=False
)

# ======================
# 4) 一个小工具：拿“最终层 CLS 向量”
# ======================
def extract_cls(feats: torch.Tensor) -> torch.Tensor:
    """
    根据 timm 不同版本返回的 forward_features 结果，稳健地取 CLS 向量：
      - 如果 forward_features 返回 dict，优先使用 'x_norm_clstoken'（已 LayerNorm 过的 CLS）
      - 如果返回 [B, T, C] 的 3D 张量，取第 0 个 token 作为 CLS（feats[:, 0]）
      - 如果返回 [B, C] 的 2D 张量，说明已经是聚合好的向量，直接返回
      - 如果返回 [B, T, C] 但没有 CLS（少见），则对 token 维度做平均
    """
    if isinstance(feats, dict):
        # 新版 timm 常见返回
        if "x_norm_clstoken" in feats and feats["x_norm_clstoken"] is not None:
            return feats["x_norm_clstoken"]         # [B, C]
        if "cls_token" in feats and feats["cls_token"] is not None:
            return feats["cls_token"]               # [B, C]
        if "last_hidden_state" in feats and feats["last_hidden_state"] is not None:
            feats = feats["last_hidden_state"]      # 兜底到 3D

    if torch.is_tensor(feats):
        if feats.ndim == 3:
            # 形状 [B, T, C]
            # 常见做法：第 0 个 token 是 CLS
            cls = feats[:, 0]                       # [B, C]
            # 有些实现没有 CLS（极少），可改成 feats.mean(dim=1)
            return cls
        if feats.ndim == 2:
            # 已是 [B, C]
            return feats

    raise RuntimeError("未能从 forward_features 的输出中解析 CLS 向量")

# ======================
# 5) 主提取函数：读图 -> 预处理 -> forward_features -> 取 CLS -> 累积
# ======================
@torch.inference_mode()
def extract_embeddings():
    files = [
        f for f in sorted(os.listdir(IMAGE_FOLDER))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"))
    ]
    if not files:
        raise RuntimeError(f"未在目录中发现图片：{IMAGE_FOLDER}")

    vecs, names = [], []

    for i in range(0, len(files), BATCH_SIZE):
        batch_files = files[i:i+BATCH_SIZE]

        # 读图并预处理到 [B,3,224,224]
        xs = []
        for fname in batch_files:
            img = Image.open(os.path.join(IMAGE_FOLDER, fname)).convert("RGB")
            x = preprocess(img).unsqueeze(0)   # [1,3,224,224]
            xs.append(x)
        x = torch.cat(xs, dim=0).to(device)    # [B,3,224,224]

        # 前向：timm 的 ViT 推荐用 forward_features 拿最后一层隐藏表示
        feats = net.forward_features(x)
        cls = extract_cls(feats)               # [B, C]，ViT-B/16 的 C=768

        f = cls.detach().cpu().numpy().astype(np.float32)
        vecs.append(f); names.extend(batch_files)

        print(f"[Batch {i//BATCH_SIZE + 1}] Processed {len(batch_files)} images")

    X = np.vstack(vecs)  # (N, 768)
    return X, names

# ======================
# 6) 保存输出
# ======================
if __name__ == "__main__":
    X, names = extract_embeddings()
    np.save(os.path.join(OUTPUT_FOLDER, "embeddings.npy"), X)
    with open(os.path.join(OUTPUT_FOLDER, "filenames.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names))
    print(f"[Done][ViT-B/16] 提取完成: X={X.shape}, 已保存到 {OUTPUT_FOLDER}")
