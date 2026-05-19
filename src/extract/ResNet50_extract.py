import os, argparse, numpy as np
from PIL import Image
import torch, torch.nn as nn
from torchvision import models

def device():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dev = device()

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    net = models.resnet50(weights=weights).eval().to(dev)
    preprocess = weights.transforms(antialias=True)
    trunk = nn.Sequential(*(list(net.children())[:-1])).eval().to(dev)  # 去掉 fc

    files = [f for f in sorted(os.listdir(args.images)) if f.lower().endswith((".jpg",".jpeg",".png"))]
    vecs, names = [], []

    with torch.inference_mode():
        for i in range(0, len(files), args.batch):
            batch = files[i:i+args.batch]
            xs = [preprocess(Image.open(os.path.join(args.images, f)).convert("RGB")).unsqueeze(0) for f in batch]
            x  = torch.cat(xs, 0).to(dev)
            f  = torch.flatten(trunk(x), 1).cpu().numpy().astype(np.float32)
            vecs.append(f); names.extend(batch)

    X = np.vstack(vecs)
    np.save(os.path.join(args.outdir, "embeddings.npy"), X)
    with open(os.path.join(args.outdir, "filenames.txt"), "w") as f: f.write("\n".join(names))
    print(f"[Done][ResNet50] X={X.shape} -> {args.outdir}")

if __name__ == "__main__":
    main()
-