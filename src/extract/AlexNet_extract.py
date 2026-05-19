import os
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torchvision import models

IMAGE_FOLDER  = "/Users/kristin/Desktop/vgg_project/Prady_art_dataset"
OUTPUT_FOLDER = "/Users/kristin/Desktop/vgg_new/AlexNet-embeddings"
BATCH_SIZE    = 16
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using {device}")

weights    = models.AlexNet_Weights.IMAGENET1K_V1
net        = models.alexnet(weights=weights).eval().to(device)
preprocess = weights.transforms(antialias=True)

# features → GAP → Flatten → [B,256]
trunk = nn.Sequential(
    net.features,
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten()
).eval().to(device)

@torch.inference_mode()
def extract_embeddings():
    files = [f for f in sorted(os.listdir(IMAGE_FOLDER))
             if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp",".tiff"))]
    vecs, names = [], []
    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i:i+BATCH_SIZE]
        xs = [preprocess(Image.open(os.path.join(IMAGE_FOLDER, f)).convert("RGB")).unsqueeze(0) for f in batch]
        x  = torch.cat(xs, 0).to(device)
        f  = trunk(x).cpu().numpy().astype(np.float32)  # [B,256]
        vecs.append(f); names.extend(batch)
        print(f"[Batch {i//BATCH_SIZE+1}] Processed {len(batch)} images")
    X = np.vstack(vecs)                                  # (N,256)
    return X, names

if __name__ == "__main__":
    X, names = extract_embeddings()
    np.save(os.path.join(OUTPUT_FOLDER, "embeddings.npy"), X)
    with open(os.path.join(OUTPUT_FOLDER, "filenames.txt"), "w") as f: f.write("\n".join(names))
    print(f"[Done][AlexNet] 提取完成: X={X.shape}, 已保存到 {OUTPUT_FOLDER}")
