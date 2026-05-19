"""
extract_dinov2_vitb14.py  —  DINOv2 ViT-B/14 写死路径 + 详细注释
输出：
  - embeddings.npy: (N, 768)  每张图的 CLS 向量
  - filenames.txt : N 行文件名，与 embeddings 一一对应
"""

import os
import numpy as np
from PIL import Image
import torch
import timm

# ===== 1) 路径与批大小（写死） =====
IMAGE_FOLDER  = "/Users/kristin/Desktop/vgg_project/Prady_art_dataset"
OUTPUT_FOLDER = "/Users/kristin/Desktop/vgg_new/DINOv2-B14-embeddings"
BATCH_SIZE    = 4   # ⚠️ DINOv2 输入 518×518 很大，建议先用 4 或 2，避免 OOM
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===== 2) 设备选择 =====
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using {device}")

# ===== 3) 创建 DINOv2 模型 + 自动匹配预处理尺寸 =====
model_name = "vit_base_patch14_dinov2"
net = timm.create_model(model_name, pretrained=True)
net.eval().to(device)

# 从模型配置读取期望的输入尺寸（DINOv2 常为 (3, 518, 518)）
input_size = net.pretrained_cfg.get("input_size", (3, 224, 224))
print(f"[Model] {model_name} expects input_size={input_size}")

preprocess = timm.data.transforms_factory.create_transform(
    input_size=input_size,   # ✅ 自动适配 518 或 224
    is_training=False
)

# ===== 4) 取 CLS 向量的工具函数（兼容 timm 不同返回格式） =====
def extract_cls(feats: torch.Tensor) -> torch.Tensor:
    """
    - 如果 forward_features 返回 dict：优先 'x_norm_clstoken'，否则 'cls_token'，否则 'last_hidden_state'
    - 如果是 [B,T,C]：取 feats[:,0] 作为 CLS
    - 如果是 [B,C]：直接返回
    - 兜底：对 tokens 取平均（很少用到）
    """
    if isinstance(feats, dict):
        if "x_norm_clstoken" in feats and feats["x_norm_clstoken"] is not None:
            return feats["x_norm_clstoken"]       # [B, C]
        if "cls_token" in feats and feats["cls_token"] is not None:
            return feats["cls_token"]             # [B, C]
        if "last_hidden_state" in feats and feats["last_hidden_state"] is not None:
            feats = feats["last_hidden_state"]    # → [B, T, C]

    if torch.is_tensor(feats):
        if feats.ndim == 3:
            return feats[:, 0]                    # [B, C]，第 0 个 token 当 CLS
        if feats.ndim == 2:
            return feats                          # [B, C]

    # 兜底（极少触发）
    return feats.mean(dim=1)

# ===== 5) 主提取流程 =====
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

        # 读图 & 预处理到 [B, 3, H, W]，其中 (H,W) 来自 input_size（通常为 518×518）
        xs = []
        for fname in batch_files:
            img = Image.open(os.path.join(IMAGE_FOLDER, fname)).convert("RGB")
            x = preprocess(img).unsqueeze(0)          # [1, 3, H, W]
            xs.append(x)
        x = torch.cat(xs, dim=0).to(device)           # [B, 3, H, W]

        # 前向：拿到最后一层隐藏状态
        feats = net.forward_features(x)

        # 取 CLS 向量（[B, 768]）
        cls = extract_cls(feats)

        # 存到 numpy
        f = cls.detach().cpu().numpy().astype(np.float32)
        vecs.append(f); names.extend(batch_files)

        print(f"[Batch {i//BATCH_SIZE + 1}] Processed {len(batch_files)} images")

    X = np.vstack(vecs)    # (N, 768)
    return X, names

# ===== 6) 保存产物 =====
if __name__ == "__main__":
    X, names = extract_embeddings()
    np.save(os.path.join(OUTPUT_FOLDER, "embeddings.npy"), X)
    with open(os.path.join(OUTPUT_FOLDER, "filenames.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names))
    print(f"[Done][DINOv2 ViT-B/14] 提取完成: X={X.shape}, 已保存到 {OUTPUT_FOLDER}")
