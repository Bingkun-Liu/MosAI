import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

# ====== 路径设置 ======
image_folder = "/Users/kristin/Desktop/vgg_project/Prady_art_dataset"
output_folder = "/Users/kristin/Desktop/vgg_new/VGG16-embeddings"
os.makedirs(output_folder, exist_ok=True)

# ====== 使用官方权重与官方 transforms（更稳、更可复现） ======
# 说明：这一步替代了你原来的手写 Compose(Resize(224,224)+Normalize)，
# 官方 transforms 默认是：Resize(短边=256，保持长宽比) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet mean/std)。
# 标准化后的像素：原像素是 0-255（整数），ToTensor 后变 0-1（浮点），再按通道做 Normalize。
weights = models.VGG16_Weights.IMAGENET1K_V1
preprocess = weights.transforms(antialias=True)

# ====== 加载完整 VGG16，并构造“主干+GAP+Flatten”得到 512 维向量 ======
# vgg.features -> AdaptiveAvgPool2d(1) -> Flatten
# 等价于取“最后一个卷积块（conv5_3+relu5_3 之后，再经过第5次 maxpool）输出 [B,512,7,7]”，
# 再做 GAP 变成 [B,512,1,1]，Flatten 成 [B,512]，作为统一“语义向量位点”。
vgg = models.vgg16(weights=weights).eval()
backbone = nn.Sequential(
    vgg.features,                  # 输出张量形状 [B, 512, 7, 7]（最后 maxpool 之后）
    nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling: [B, 512, 1, 1]
    nn.Flatten()                   # -> [B, 512]
).eval()


# 增加 mps 优先，其次 cuda，否则 cpu
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
backbone.to(device)

# ====== （已移除）L2 标准化函数 ======
# [修改] 为了让所有模型在“回归阶段 Pipeline”里统一做 L2，这里不再做样本级 L2
# 如需保留函数也可以，但不要在本脚本中调用
# def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
#     norms = np.linalg.norm(X, ord=2, axis=1, keepdims=True)
#     return X / np.maximum(norms, eps)

# ====== 主流程：遍历图片，提取 512 维向量 ======
embeds = []
names = []

with torch.inference_mode():  # 等价于 no_grad，但推理更优化
    for filename in sorted(os.listdir(image_folder)):  # 用 sorted 保证顺序稳定
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(image_folder, filename)
        img = Image.open(path).convert("RGB")

        # 官方预处理（Resize(256, 保例) -> CenterCrop(224) -> ToTensor -> Normalize）
        x = preprocess(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

        # 一次前向 -> 直接得到 [1, 512] 的“最后层语义向量”
        feat_512 = backbone(x)                       # [1, 512]
        embeds.append(feat_512.cpu().numpy()[0].astype(np.float32))  # (512,)
        names.append(filename)

X = np.vstack(embeds).astype(np.float32)   # (N, 512)

# 保存结果
np.save(os.path.join(output_folder, "embeddings.npy"), X)
with open(os.path.join(output_folder, "filenames.txt"), "w") as f:
    f.write("\n".join(names))

print(f"Saved embeddings to {os.path.join(output_folder, 'embeddings.npy')} with shape {X.shape}")
print(f"Saved filenames to   {os.path.join(output_folder, 'filenames.txt')}")
