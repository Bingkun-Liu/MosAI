#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化易懂版：统一划分 + TEST 排名（分数/误差）+ HTML 图墙（页面内弹窗预览）
-----------------------------------------------------------------
要点不变：
- 与你回归一致：Normalizer('l2') -> StandardScaler() -> RidgeCV(logspace(-4,6,60))
- 所有模型共用同一份 train/test 划分（在“模型都存在 + 有效标签”的交集上）
- 只对 TEST 预测与排序（避免评估污染），导出 CSV + HTML（可点击预览）
"""

import os, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image, ImageOps

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ======================== 用户配置区（改这里） ========================

IMAGE_FOLDER  = "/Users/kristin/Desktop/vgg_project/Prady_art_dataset"

MODEL_DIRS: Dict[str, str] = {
    "alexnet":     "/Users/kristin/Desktop/vgg_shuffled 2/AlexNet-embeddings",
    "git_base":    "/Users/kristin/Desktop/vgg_shuffled 2/GIT-embeddings",
    "blip2":       "/Users/kristin/Desktop/vgg_shuffled 2/BLIP2-embeddings",
    "clip":        "/Users/kristin/Desktop/vgg_shuffled 2/CLIP-B32-embeddings",
    "ResNet50":    "/Users/kristin/Desktop/vgg_shuffled 2/ResNet50-embeddings",
    "InceptionV3": "/Users/kristin/Desktop/vgg_shuffled 2/Inception-embeddings",
    "VGG16":       "/Users/kristin/Desktop/vgg_shuffled 2/VGG16-embeddings",
    "ViT-B16":     "/Users/kristin/Desktop/vgg_shuffled 2/ViT-B16-embeddings",
    "SigLIP2":     "/Users/kristin/Desktop/vgg_shuffled 2/SigLIP2-embeddings",
    "ImageBind":   "/Users/kristin/Desktop/vgg_shuffled 2/ImageBind-embeddings",
    "Clip2":       "/Users/kristin/Desktop/vgg_shuffled 2/CLIP-B16-embeddings",
    "DINOv2":      "/Users/kristin/Desktop/vgg_shuffled 2/DINOv2-B14-embeddings",
    "Florence2":  "/Users/kristin/Desktop/vgg_shuffled 2/Florence2-embeddings",



}

MAT_PATH  = "/Users/kristin/Desktop/vgg_shuffled 2/image_mean_rating_shuffled2.mat"
OUT_DIR   = "/Users/kristin/Desktop/vgg_shuffled 2/basic_html_shared_test_original"

TEST_SIZE = 0.2     # 测试集比例（20%）
SEED      = 42      # 随机种子（可复现）

# HTML 用到的图片尺寸（导出到 OUT_DIR 下的 thumbnails/ 和 previews/）
THUMB_SIZE    = (256, 256)     # 卡片小图
PREVIEW_SIZE  = (1024, 1024)   # 弹窗大图
JPEG_QUALITY  = 88
# 扫描图片时允许的扩展名
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ======================== 下面是程序主体 ========================

def build_pipeline() -> Pipeline:
    """与统一回归一致的三步流水线。"""
    return Pipeline([
        ("l2",     Normalizer(norm="l2")),
        ("scaler", StandardScaler()),
        ("ridge",  RidgeCV(alphas=np.logspace(-4, 6, 60)))
    ])

def ensure_dir(p: Path): # 创建文件夹
    p.mkdir(parents=True, exist_ok=True)

# ----- 第 1 步：读取标签（MAT），输出 {stem -> score}，并过滤 NaN/Inf -----
def load_label_map_from_mat(mat_path: str) -> Dict[str, float]:
    mat = sio.loadmat(mat_path)
    # image_names 取主干名（去扩展名），常见形状是 (1, N)
    stems = [os.path.splitext(str(x[0]).strip())[0] for x in mat["image_names"][0]]
    if "mean_score" in mat:
        scores = mat["mean_score"].flatten().astype(np.float32)
    elif "mean_rating" in mat:
        scores = mat["mean_rating"].flatten().astype(np.float32)
    else:
        raise ValueError("MAT 缺少 mean_score/mean_rating")

    if len(stems) != len(scores):
        raise ValueError("MAT image_names 与 scores 数量不一致")

    m: Dict[str, float] = {}
    bad = 0
    for s, v in zip(stems, scores):
        if np.isfinite(v):
            m[s] = float(v)
        else:
            bad += 1
    if bad:
        print(f"[Clean] 过滤掉 {bad} 个 NaN/Inf 标签")
    return m

# 读取 embeddings 与文件名
#取主干名对齐用 → 过滤 NaN/Inf→ 返回字典 {主干名: 分数}
def load_embeddings(model_dir: str) -> Tuple[np.ndarray, List[str]]: # 返回类型是 Dict[str, float]（键：字符串，值：浮点）ex ：{ "dog_001": 3.75, "cat_002": 4.10, ... }
    X_path  = os.path.join(model_dir, "embeddings.npy")
    fn_path = os.path.join(model_dir, "filenames.txt")
    if not (os.path.exists(X_path) and os.path.exists(fn_path)):
        raise FileNotFoundError(f"缺少 embeddings.npy 或 filenames.txt 于 {model_dir}")
    X = np.load(X_path).astype(np.float32)
    with open(fn_path, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if X.shape[0] != len(names):
        raise ValueError(f"[{model_dir}] 行数不一致: X={X.shape[0]} vs names={len(names)}")
    return X, names

# ----- 第 3 步：为每个模型建立“主干名 -> 行号/原文件名”的映射，便于交集对齐 -----
def names_to_maps(names: List[str]) -> Tuple[Dict[str, int], Dict[str, str]]:
    stem2idx, stem2name = {}, {}
    for i, fn in enumerate(names):
        st = Path(fn).stem
        if st not in stem2idx:        # 有重复主干名时保留第一个
            stem2idx[st]  = i
            stem2name[st] = fn
    return stem2idx, stem2name

# ----- 第 4 步：扫描图片库，做简单索引（为了 HTML 能找到原图，不走 file://） -----
def scan_images(root: Path) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """
    返回两个索引（小写）：
      by_basename: 'xxx.jpg' -> [路径...]
      by_stem:     'xxx'     -> [路径...]
    """
    by_basename, by_stem = {}, {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            base = p.name.lower()
            st   = p.stem.lower()
            by_basename.setdefault(base, []).append(p)
            by_stem.setdefault(st, []).append(p)
    return by_basename, by_stem

def resolve_image_path(image_folder: Path,
                       by_basename: Dict[str, List[Path]],
                       by_stem: Dict[str, List[Path]],
                       fn: str) -> Optional[Path]:
    """尽量把 filenames.txt 里的名字映射到真实图片路径（大小写不敏感；可用主干名匹配）。"""
    p = Path(fn)
    # 绝对路径
    if p.is_absolute() and p.exists():
        return p
    # 相对路径：IMAGE_FOLDER / fn
    direct = image_folder / fn
    if direct.exists():
        return direct
    # 名称大小写不敏感
    base = p.name.lower()
    if base in by_basename and by_basename[base]:
        return by_basename[base][0]
    # 主干名匹配
    st = p.stem.lower()
    if st in by_stem and by_stem[st]:
        return by_stem[st][0]
    return None

# ----- 第 5 步：把原图导出为缩略图/预览图（JPEG），供 HTML 使用 -----
def save_sized_image(src: Path, dst: Path, size=(256,256), quality=88) -> bool:
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im = ImageOps.pad(im, size, method=Image.Resampling.LANCZOS, color=(245,245,245))
            ensure_dir(dst.parent)
            im.save(dst, format="JPEG", quality=quality, optimize=True, progressive=True)
            return True
    except Exception as e:
        print(f"[Image ERR] {src}: {e}")
        return False

# ----- 第 6 步：写 HTML（非常直白一个函数：给定标题和 DataFrame -> 返回 HTML 字符串） -----
def make_gallery_html(title: str,
                      df: pd.DataFrame,
                      image_folder: Path,
                      out_dir: Path,
                      by_basename: Dict[str, List[Path]],
                      by_stem: Dict[str, List[Path]],
                      thumbs_dir: Path,
                      previews_dir: Path) -> str:
    """df 必须包含列：rank, filename, score（score 只是展示文本）"""
    cards = []
    for _, row in df.iterrows():
        fn = str(row["filename"])
        score = float(row["score"])
        real = resolve_image_path(image_folder, by_basename, by_stem, fn)

        # 导出资源文件名（统一按主干名命名）
        stem = Path(fn).stem
        thumb = thumbs_dir / f"{stem}.jpg"
        prev  = previews_dir / f"{stem}.jpg"
        thumb_rel, prev_rel = "", ""

        if real and real.exists():
            if not thumb.exists():
                save_sized_image(real, thumb, size=THUMB_SIZE, quality=JPEG_QUALITY)
            if not prev.exists():
                save_sized_image(real, prev, size=PREVIEW_SIZE, quality=JPEG_QUALITY)
            if thumb.exists():
                thumb_rel = os.path.relpath(thumb, out_dir)
            if prev.exists():
                prev_rel  = os.path.relpath(prev,  out_dir)

        img_html = f'<img src="{thumb_rel}" alt="{fn}" loading="lazy" />' if thumb_rel else '<div class="ph">Not Found</div>'
        onclick  = f"openViewer('{prev_rel}','{fn}','{score:.4f}')" if prev_rel else "alert('预览图不存在');"

        card = f"""
        <div class="card" onclick="{onclick}">
          <div class="rk">#{int(row['rank'])}</div>
          {img_html}
          <div class="meta">
            <div class="fn" title="{fn}">{fn}</div>
            <div class="score">score: {score:.4f}</div>
          </div>
        </div>
        """
        cards.append(card)

    html = f"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>
body {{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;background:#fafafa}}
.wrap {{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}}
.card {{cursor:pointer;border:1px solid #e5e7eb;border-radius:14px;padding:10px;background:#fff;box-shadow:0 2px 6px rgba(0,0,0,.06);position:relative}}
.card:hover {{box-shadow:0 6px 16px rgba(0,0,0,.12);transform:translateY(-2px);transition:.15s}}
.card img,.ph {{width:100%;height:180px;object-fit:cover;border-radius:10px;background:#f3f4f6;display:block}}
.ph {{display:flex;align-items:center;justify-content:center;color:#9ca3af}}
.meta {{margin-top:8px;display:flex;justify-content:space-between;gap:8px}}
.fn {{font-size:12px;color:#374151;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:70%}}
.score {{font-size:12px;color:#6b7280}}
.rk {{position:absolute;top:8px;left:8px;background:#111827;color:#fff;font-size:12px;padding:2px 6px;border-radius:999px}}
#viewer {{display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);align-items:center;justify-content:center;z-index:9999}}
#viewer img {{max-width:90vw;max-height:85vh;border-radius:12px}}
#viewer .info {{color:#e5e7eb;margin-top:10px;text-align:center;font-size:13px}}
#viewer .close {{position:absolute;top:16px;right:16px;background:#111827;color:#fff;border:none;border-radius:999px;padding:8px 12px;cursor:pointer}}
.top {{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}}
.badge {{font-size:12px;color:#6b7280}}
</style>
<body>
  <div class="top">
    <h1>{title}</h1>
    <div class="badge">统一划分，共用 TEST；点击卡片预览</div>
  </div>
  <div class="wrap">
    {''.join(cards)}
  </div>
  <div id="viewer">
    <button class="close" onclick="closeViewer()">关闭</button>
    <div>
      <img id="big" src="" alt="preview"><div id="info" class="info"></div>
    </div>
  </div>
<script>
function openViewer(src, fn, score) {{
  if (!src) {{ alert('预览图不存在'); return; }}
  document.getElementById('big').src = src;
  document.getElementById('info').textContent = fn + ' | score: ' + score;
  document.getElementById('viewer').style.display = 'flex';
}}
function closeViewer() {{
  document.getElementById('viewer').style.display = 'none';
  document.getElementById('big').src = '';
}}
</script>
</body>
</html>
"""
    return html

def main():
    # 目录准备
    out_dir = Path(OUT_DIR); ensure_dir(out_dir)
    thumbs  = out_dir / "thumbnails"; ensure_dir(thumbs)
    prevs   = out_dir / "previews";   ensure_dir(prevs)

    # 扫描图片（为了 HTML 找图，首次可能稍慢）
    print("[Index] 扫描图片库 ...")
    image_folder = Path(IMAGE_FOLDER)
    by_basename, by_stem = scan_images(image_folder)
    print(f"[Index] 完成：basename={len(by_basename)} / stem={len(by_stem)}")

    # 读取标签：{stem -> score}
    label_map = load_label_map_from_mat(MAT_PATH)

    # 读取每个模型的 X 与 names，并建立 stem 映射
    model_data: Dict[str, Dict] = {}
    for name, d in MODEL_DIRS.items():
        print(f"\n=== Load {name} ===")
        X, names = load_embeddings(d)
        s2i, s2n = names_to_maps(names)
        model_data[name] = dict(X=X, stem2idx=s2i, stem2name=s2n)
        print(f"[{name}] X={X.shape}, unique_stems={len(s2i)}")

    # 求交集（所有模型都覆盖 且 有有效标签的主干名）
    common = set(label_map.keys())
    for md in model_data.values():
        common &= set(md["stem2idx"].keys())
    stems = sorted(common)
    if len(stems) < 3:
        raise RuntimeError(f"交集太小：{len(stems)}")

    # 与交集顺序对齐的 y
    y_all = np.array([label_map[s] for s in stems], dtype=np.float32)

    # 一次统一划分（所有模型共用）
    idx_all = np.arange(len(stems))
    tr_idx, te_idx = train_test_split(idx_all, test_size=TEST_SIZE, random_state=SEED)
    print(f"\n[Shared Split] train={len(tr_idx)}  test={len(te_idx)}  common={len(stems)}")

    # 保存划分（可复现）
    split_info = {
        "TEST_SIZE": TEST_SIZE,
        "SEED": SEED,
        "stems_common": stems,
        "train_stems": [stems[i] for i in tr_idx],
        "test_stems":  [stems[i] for i in te_idx],
    }
    (out_dir / "shared_split.json").write_text(json.dumps(split_info, ensure_ascii=False, indent=2), encoding="utf-8")

    # 逐模型训练/预测（只在 TEST 上出结果）
    summary_rows = []
    index_links  = []  # (标题, 路径) 供主页链接

    for model_name, md in model_data.items():
        print(f"\n=== Train/Test {model_name} ===")
        X = md["X"]; s2i = md["stem2idx"]; s2n = md["stem2name"]

        # 对齐该模型的特征到“交集顺序”
        idxs = [s2i[s] for s in stems]
        X_all = X[np.array(idxs, dtype=int)]
        names_ordered = np.array([s2n[s] for s in stems])

        # 划分同一组 train/test
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_te, y_te = X_all[te_idx], y_all[te_idx]
        names_te   = names_ordered[te_idx]

        # 训练 + 预测（只在 TEST 上评估/排序）
        pipe = build_pipeline()
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te).astype(np.float32)

        # 指标
        mse = mean_squared_error(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)
        r2  = r2_score(y_te, y_pred)
        print(f"[{model_name}][Test] MSE={mse:.6f} MAE={mae:.6f} R2={r2:.6f}")
        summary_rows.append(dict(model=model_name, n_train=len(tr_idx), n_test=len(te_idx),
                                 MSE=mse, MAE=mae, R2=r2))

        # ========== A) 测试集“按预测分数”降序 ==========
        order = np.argsort(-y_pred)
        ranked_df = pd.DataFrame({
            "rank":     np.arange(1, len(order)+1),
            "filename": names_te[order],
            "score":    y_pred[order].astype(np.float32),
        })
        ranked_csv = out_dir / f"ranking_{model_name}_TEST.csv"
        ranked_df.to_csv(ranked_csv, index=False)

        rank_html_title = f"{model_name} — TEST 排名（按预测分数）"
        rank_html_path  = out_dir / f"gallery_{model_name}_TEST_score.html"
        rank_html       = make_gallery_html(rank_html_title, ranked_df, image_folder, out_dir,
                                            by_basename, by_stem, thumbs, prevs)
        rank_html_path.write_text(rank_html, encoding="utf-8")
        index_links.append((rank_html_title, rank_html_path))

        # ========== B) 测试集“按平方误差”升序（越前越准）==========
        sq_err = (y_te - y_pred) ** 2
        err_order = np.argsort(sq_err)  # 小误差排前
        err_df = pd.DataFrame({
            "rank":     np.arange(1, len(err_order)+1),
            "filename": names_te[err_order],
            "score":    (-sq_err[err_order]).astype(np.float32),  # 为了 HTML 视觉，score 用负误差（越大越准）
            "y_true":   y_te[err_order].astype(np.float32),
            "y_pred":   y_pred[err_order].astype(np.float32),
            "sq_err":   sq_err[err_order].astype(np.float32),
        })
        err_csv = out_dir / f"errors_{model_name}_TEST.csv"
        err_df.to_csv(err_csv, index=False)

        err_html_title = f"{model_name} — TEST 排名（按误差：越前越准）"
        err_html_path  = out_dir / f"gallery_{model_name}_TEST_error.html"
        err_html       = make_gallery_html(err_html_title, err_df[["rank","filename","score"]],
                                           image_folder, out_dir, by_basename, by_stem, thumbs, prevs)
        err_html_path.write_text(err_html, encoding="utf-8")
        index_links.append((err_html_title, err_html_path))

    # 汇总指标 & 主页
    summ = pd.DataFrame(summary_rows).sort_values("R2", ascending=False)
    summ_path = out_dir / "summary_metrics_TEST.csv"
    summ.to_csv(summ_path, index=False)

    index_html = out_dir / "index.html"
    links = "\n".join([f'<li><a href="{p.name}">{title}</a></li>' for title, p in index_links])
    index_html.write_text(
        f"<!doctype html><meta charset='utf-8'><title>Model TEST Galleries</title>"
        f"<h1>Model TEST Galleries</h1>"
        f"<p>统一划分：train={len(tr_idx)} test={len(te_idx)} common={len(stems)}</p>"
        f"<p>指标汇总：<a href='summary_metrics_TEST.csv'>summary_metrics_TEST.csv</a></p>"
        f"<ul>{links}</ul>",
        encoding="utf-8"
    )
    print(f"\n[Index] {index_html}")
    print("完成：每个模型已生成 2 个 HTML（分数/误差）+ 2 个 CSV（ranking/errors）+ 指标汇总。")

if __name__ == "__main__":
    main()
