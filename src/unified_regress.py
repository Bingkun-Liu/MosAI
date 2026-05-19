#!/usr/bin/env python
# ===== 超详细注释版：只做回归 =====
# 目标：把已提取好的 embedding（X）和评分（y）对齐、清理，然后训练一个稳定的回归模型并评估

import os
import numpy as np
import scipy.io as sio   # 专门用来读 .mat（Matlab）文件

# --- 这几类是机器学习常用工具 ---
from sklearn.pipeline import Pipeline             # 把多个步骤串成一条流水线
from sklearn.preprocessing import Normalizer      # L2 归一化（按“每一行/每个样本”缩放到单位长度）
from sklearn.preprocessing import StandardScaler  # 标准化（按“每一列/每个特征维度”做零均值、单位方差）
from sklearn.linear_model import RidgeCV          # 岭回归 + 内置交叉验证自动选正则强度
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========= 1) 路径和基本参数（你只要改这里） =========
# EMB_DIR：embedding 的目录 —— 里面必须有两个文件：
#   - embeddings.npy  （形状 (N, D)，N=样本数，D=特征维度，比如 512）
#   - filenames.txt   （N 行文本，每行一个文件名，与 embeddings.npy 的行一一对应）
EMB_DIR   = "/Users/kristin/Desktop/vgg_shuffled 2/Florence2-embeddings"

# MAT_PATH：MATLAB 的标签文件 —— 里面要有：
#   - image_names     （图片名列表）
#   - mean_score 或 mean_rating （每张图的评分）
MAT_PATH  = "/Users/kristin/Desktop/vgg_shuffled 2/image_mean_rating_shuffled2.mat"


# 训练/评估的基本超参数
TEST_SIZE = 0.2   # 测试集比例：0.2 = 20%
SEED      = 42    # 随机种子：保证每次划分一致，结果可复现
CV_FOLDS  = 5     # 交叉验证折数；设为 0 表示不做交叉验证


# ========= 2) 读取 embeddings 和文件名 =========
def load_embeddings(emb_dir: str):
    """
    读取特征矩阵 X 与文件名 names。
    返回：
      X: np.ndarray, 形状 (N, D)
      names: list[str], 长度 N
    """
    X_path = os.path.join(emb_dir, "embeddings.npy")
    fn_path= os.path.join(emb_dir, "filenames.txt")

    # 基本的文件存在性检查，避免后面读不到再报一堆栈
    if not os.path.exists(X_path) or not os.path.exists(fn_path):
        raise FileNotFoundError(f"缺少 embeddings.npy 或 filenames.txt 于 {emb_dir}")

    # 读特征：得到一个二维数组，N 行（图片数），每行 D 个数字（向量维度）
    X = np.load(X_path).astype(np.float32)  # 统一用 float32 足够且省内存

    # 读文件名：每行一个，strip 去掉换行符
    with open(fn_path, "r") as f:
        names = [ln.strip() for ln in f if ln.strip()]

    # 核对长度：X 的行数必须和 names 的个数一致
    if len(names) != len(X):
        raise ValueError(f"embeddings 行数({len(X)}) 与 filenames 数量({len(names)}) 不一致")

    print(f"[Load] X={X.shape}, filenames={len(names)}")
    # 举例：X.shape = (826, 512) 表示有 826 张图，每张 512 维向量
    return X, names


# ========= 3) 从 .mat 文件读取并对齐标签 =========
def load_labels_from_mat_like_clip(mat_path: str, filenames_txt):
    """
      - .mat 中的 image_names：取“主干名”（去掉扩展名）
      - 把 .mat 的 (主干名 -> 分数) 做成字典
      - 把 filenames.txt 也转成主干名，一一去字典里找分数
      - 找不到的样本跳过
      - 如果分数是 NaN，也跳过
    返回：
      y: np.ndarray, 形状 (M,)  —— 对齐后的标签（M <= N）
      keep_idx: np.ndarray, 形状 (M,) —— 原始列表中保留下来的下标（用于把 X 同步筛选成 M 行）
    """
    # 读 .mat：得到一个字典，键是变量名，值是 numpy 数组
    mat = sio.loadmat(mat_path)

    # 1) 取 .mat 里的图片名 “主干”（不带扩展名）
    #    这行和你 CLIP 的写法一模一样，通常 image_names 的形状是 (1, N)
    image_names = [os.path.splitext(str(x[0]).strip())[0] for x in mat["image_names"][0]]

    # 2) 取评分数组（兼容 mean_score 或 mean_rating 两个名字）
    if "mean_score" in mat:
        scores = mat["mean_score"].flatten().astype(np.float32)
    elif "mean_rating" in mat:
        scores = mat["mean_rating"].flatten().astype(np.float32)
    else:
        raise ValueError("MAT 文件缺少 mean_score/mean_rating")

    # 基本一致性检查
    if len(image_names) != len(scores):
        raise ValueError("MAT 内部 image_names 与 scores 数量不一致")

    # 3) 建立字典：<主干名> -> 分数
    score_map = dict(zip(image_names, scores))

    # 4) 把 filenames.txt 也转成主干名，然后按顺序去字典里查分数
    #    注意：为了防止个别图片（比如 *_副本）在 .mat 里没有分数，我们做“可跳过”的逻辑
    names_base = [os.path.splitext(n)[0] for n in filenames_txt]

    y = []           # 存放有效的分数
    keep_idx = []    # 存放“哪些行保留”的下标（用于把 X 一起筛选）
    for i, (b, fn) in enumerate(zip(names_base, filenames_txt)):
        if b in score_map:
            y.append(score_map[b])
            keep_idx.append(i)
        else:
            # 这里就是你之前遇到的 *_副本 的情况：在 .mat 里没有原图的评分
            print(f"[Skip] {fn} 在 MAT 中找不到分数，跳过")

    # 先转成 numpy 数组
    y = np.asarray(y, dtype=np.float32)
    keep_idx = np.asarray(keep_idx, dtype=np.int64)

    # 5) 清理 NaN：有些分数可能缺失（NaN），我们继续把它们剔除
    mask = ~np.isnan(y)     # True 表示这个位置有效（不是 NaN）
    if not np.all(mask):    # 如果存在 NaN
        print(f"[Clean] 检测到 {np.sum(~mask)} 个 NaN 标签，已跳过")
        y = y[mask]                 # 只保留有效分数
        keep_idx = keep_idx[mask]   # X 的下标也同步保留有效的

    print(f"[Align] 对齐完成：y.shape={y.shape}, 保留 {len(keep_idx)} / {len(filenames_txt)}")
    # 举例输出：y.shape=(824,) 表示最终有 824 个样本有效
    return y, keep_idx


# ========= 4) 定义“统一回归流水线” =========
def build_pipeline():
    """
    返回一个 sklearn 的 Pipeline（流水线）：
      1) Normalizer(norm="l2")    —— 对“每个样本”做 L2 归一化（把整行缩放到长度=1）
      2) StandardScaler()         —— 对“每个特征维度”做标准化（列方向：0均值、1方差），仅在训练集上 fit
      3) RidgeCV(...)             —— 岭回归，并在内部用交叉验证选择最优正则超参数 alpha
    这么做的好处：不同模型/不同特征的量纲和尺度被统一，回归更稳、更可比。
    """
    return Pipeline([
        ("l2",     Normalizer(norm="l2")),
        ("scaler", StandardScaler()),
        ("ridge",  RidgeCV(alphas=np.logspace(-4, 6, 60)))
    ])


# ========= 5) 主流程：读取 → 对齐清理 → 训练评估 =========
def main():
    # (a) 读取特征和文件名
    X, names = load_embeddings(EMB_DIR)
    # 现在：X.shape = (N, D)，names 长度 N

    # (b) 从 .mat 对齐评分，并返回 keep_idx（哪些样本有效）
    y, keep_idx = load_labels_from_mat_like_clip(MAT_PATH, names)

    # 用 keep_idx 把 X 和 names 同步筛选成“只包含有效样本”的子集
    X = X[keep_idx]                         # 形状变为 (M, D)
    names = [names[i] for i in keep_idx]    # 长度变为 M
    print(f"[Filter] X={X.shape}, y={y.shape}, names={len(names)}")

    # (c) 划分训练集/测试集
    #     注意：StandardScaler 只会在训练集上 fit（这就是 sklearn 的 Pipeline 自动做的防泄漏）
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    # (d) 建流水线并训练
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)     # X_tr = 训练集的特征；内部会先做 L2，再做标准化（fit 统计量只来自训练集），最后训练 Ridge

    # (e) 在测试集上评估
    pred = pipe.predict(X_te)
    mse = mean_squared_error(y_te, pred)   # 平方误差（越小越好）
    mae = mean_absolute_error(y_te, pred)  # 绝对误差（越小越好）
    r2  = r2_score(y_te, pred)             # 决定系数 R^2（越接近 1 越好）
    print(f"[Test] MSE={mse:.6f}  MAE={mae:.6f}  R2={r2:.6f}")

    # (f) 可选：在训练集上做 K 折交叉验证，观察平均表现和波动
    if CV_FOLDS and CV_FOLDS > 0:
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        # 注意：这里对 pipe 做 cross_val_score，意味着每个折里都会重新 fit（不会用到测试集统计量）
        cv_mse = -cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="neg_mean_squared_error")
        cv_r2  =  cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="r2")
        print(f"[CV {CV_FOLDS}折] MSE: mean={cv_mse.mean():.6f} ± {cv_mse.std():.6f}")
        print(f"[CV {CV_FOLDS}折]  R2: mean={cv_r2.mean():.6f} ± {cv_r2.std():.6f}")



# ========= 6) Python 程序入口 =========
if __name__ == "__main__":
    main()
