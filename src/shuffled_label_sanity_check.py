import scipy.io as sio
import numpy as np
import os

# 1. 读取原始 .mat 文件
mat_path = "/Users/kristin/Desktop/vgg_project/image_mean_rating.mat"
data = sio.loadmat(mat_path)

# 里面的关键变量：
# data["image_names"]        # 图片名（不动）
# data["image_categories"]   # 类别（不动）
# data["mean_score"]         # 实际打分（要打乱）

mean_score = data["mean_score"]       # 形状是 (1, 826)
n = mean_score.size                   # 元素总个数 826

# 2. 随机打乱 mean_score
rng = np.random.default_rng()         # 新版随机数生成器
# ravel() 变成一维再 permutation 打乱，最后 reshape 回原来的形状
shuffled = rng.permutation(mean_score.ravel()).reshape(mean_score.shape)

# 3. 组装新的字典并覆盖 mean_score
out_data = {k: v for k, v in data.items() if not k.startswith("__")}
out_data["mean_score"] = mean_score  #shuffled     # 用打乱后的分数替换原分数

# 4. 保存成新的 .mat 文件
out_path = "/Users/kristin/Desktop/vgg_shuffled 2/image_mean_rating_shuffled2.mat"
sio.savemat(out_path, out_data)