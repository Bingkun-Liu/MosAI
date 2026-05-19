"""
inception_v3_extract.py
=======================
目标：
  1) 遍历一个图片文件夹
  2) 用 torchvision 的 Inception v3（ImageNet 预训练）提取每张图的全局语义向量
  3) 把结果保存为：
       - embeddings.npy : 形状 (N, 2048) 的 numpy 矩阵（N=图片数）
       - filenames.txt  : N 行文本，每行是图片文件名，顺序与 embeddings 一一对应

统一协议（和你其他 CNN 保持一致）：
  - CNN 的位点统一为：最后卷积输出 → 全局平均池化(GAP, avgpool) → Flatten 得到单个向量
  - Inception v3 结构较复杂，不能直接用 children() 拼接；所以使用 feature_extractor 在 'avgpool' 处截断
  - 不在提取阶段做 L2 或 StandardScaler，这些在回归流水线里统一处理，确保跨模型可比
"""

import os
import numpy as np
from PIL import Image
import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

# ======================
# 1) 配置：路径 & 批大小（写死，便于直接运行）
# ======================
# 你的图片数据集目录（文件名会被保存到 filenames.txt 里以便对齐标签）
IMAGE_FOLDER  = "/Users/kristin/Desktop/vgg_project/Prady_art_dataset"

# 输出目录（会生成 embeddings.npy + filenames.txt）
OUTPUT_FOLDER = "/Users/kristin/Desktop/vgg_new/Inception-embeddings"

# 每个 batch 放多少张图片（太大可能显存/内存不够，太小速度慢；16 是折中）
BATCH_SIZE    = 16

# 确保输出目录存在（没有就创建）
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ======================
# 2) 选择计算设备（M1/M2 用 mps，NVIDIA 用 cuda，其他用 cpu）
# ======================
# - torch.backends.mps.is_available(): Mac Apple Silicon 的 Metal 加速
# - torch.cuda.is_available(): 是否有 NVIDIA GPU
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"[Device] Using {device}")

# ======================
# 3) 加载 Inception v3（ImageNet 预训练权重）+ 官方预处理
# ======================
# - Inception v3 的官方输入尺寸为 299x299，权重自带 transforms，确保和训练分布一致
# - 注意：不同 torchvision 版本对 aux_logits 处理不同，这里不手动改 aux_logits，使用默认构造更稳妥
weights    = models.Inception_V3_Weights.IMAGENET1K_V1
net        = models.inception_v3(weights=weights).eval().to(device)

# 官方预处理（Resize→CenterCrop（299）→ToTensor→Normalize 到 ImageNet 均值/方差）
# - antialias=True 可以让 Resize 更平滑（Pillow>=9.1）
preprocess = weights.transforms(antialias=True)

# ======================
# 4) 用 feature_extractor 在 'avgpool' 这一层把张量取出来
# ======================
# 为什么要这样做？
# - Inception v3 的 forward 不是简单的“顺序堆叠”，手动 children()[:-1] 容易打乱执行路径
# - create_feature_extractor 会按照模型原生的 forward 路径执行，
#   然后在我们指定的层（'avgpool'）把中间输出捕获出来
# 'avgpool' 的输出形状为 [B, 2048, 1, 1]，Flatten 后就是 [B, 2048] 向量
fe = create_feature_extractor(
    net,
    return_nodes={'avgpool': 'feat'}  # 键是模型里的层名，值是输出字典中的别名
).eval().to(device)

# ======================
# 5) 提取函数（核心逻辑：读图 → 预处理 → 前向 → 拿 avgpool → Flatten → 累积）
# ======================
@torch.inference_mode()  # 关闭梯度，节省内存 + 提速；与 torch.no_grad() 类似
def extract_embeddings():
    """
    遍历 IMAGE_FOLDER，分批处理，返回：
      X: (N, 2048) 的 numpy.float32 矩阵
      names: 长度 N 的文件名列表，顺序与 X 的行一一对应
    """
    # Step 5.1: 罗列所有图片文件名（按字母序排序，保证可复现/稳定）
    files = [
        f for f in sorted(os.listdir(IMAGE_FOLDER))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"))
    ]
    if not files:
        raise RuntimeError(f"在目录中未找到图片：{IMAGE_FOLDER}")

    vecs, names = [], []

    # Step 5.2: 按批处理，避免一次性塞太多图导致显存/内存不足
    for i in range(0, len(files), BATCH_SIZE):
        batch_files = files[i:i+BATCH_SIZE]

        # Step 5.3: 读图 + 预处理（Resize→CenterCrop 299→ToTensor→Normalize）
        # 预处理产物形状：[1, 3, 299, 299]；unsqueeze(0) 是为了加 batch 维度
        xs = []
        for fname in batch_files:
            img_path = os.path.join(IMAGE_FOLDER, fname)
            img = Image.open(img_path).convert("RGB")  # 统一转 RGB，避免灰度/带 alpha 的差异
            x = preprocess(img).unsqueeze(0)          # [1, 3, 299, 299]
            xs.append(x)

        # Step 5.4: 拼成一个 batch 张量：[B, 3, 299, 299]，放到 device 上
        x = torch.cat(xs, dim=0).to(device)

        # Step 5.5: 前向一次（按 Inception v3 的原生路径）
        # out 是一个 dict：{'feat': Tensor([B, 2048, 1, 1])}
        out = fe(x)
        feat_4d = out['feat']                       # [B, 2048, 1, 1]

        # Step 5.6: 全局平均池化的输出再 Flatten 到 2D：[B, 2048]
        # 为什么 Flatten？
        #   - [B, 2048, 1, 1] 的最后两个维度是“空间维度”，已经被 GAP 压到 1×1
        #   - Flatten(1) 把 [C,1,1] 展平成 [C]，得到单个 2048 维全局向量
        feat_2d = torch.flatten(feat_4d, start_dim=1)  # [B, 2048]

        # Step 5.7: 转 numpy、累积到列表里；dtype 用 float32（够用且节省空间）
        f = feat_2d.cpu().numpy().astype(np.float32)
        vecs.append(f)
        names.extend(batch_files)

        # Step 5.8: 进度提示
        print(f"[Batch {i//BATCH_SIZE + 1}] Processed {len(batch_files)} images")

    # Step 5.9: 把所有批次拼起来，得到最终矩阵 X：(N, 2048)
    X = np.vstack(vecs)
    return X, names

# ======================
# 6) 主流程（调用提取 → 保存产物）
# ======================
if __name__ == "__main__":
    # 6.1 提取
    X, names = extract_embeddings()   # X: (N, 2048), names: N 个文件名

    # 6.2 保存向量矩阵
    np.save(os.path.join(OUTPUT_FOLDER, "embeddings.npy"), X)

    # 6.3 保存文件名列表（保证行与向量一一对应，供回归阶段按名字对齐标签）
    with open(os.path.join(OUTPUT_FOLDER, "filenames.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names))

    # 6.4 完成提示
    print(f"[Done][Inception v3] 提取完成: X={X.shape}, 已保存到 {OUTPUT_FOLDER}")
