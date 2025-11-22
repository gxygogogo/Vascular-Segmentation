#%%
import os
import glob
import numpy as np
from collections import Counter

train_dir = "/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/HML_model/data_enh/HML_train"

# 找到所有 .npz 文件
npz_files = glob.glob(os.path.join(train_dir, "*.npz"))

# counter：键是类别 0/1/2，值是 patch 数量
counter = Counter()
# classes：用于记录实际出现过哪些类别
classes = set()

for fpath in npz_files:
    data = np.load(fpath)
    lbl  = data["lbl"]                # shape [H,W], 值 ∈ {0,1,2}
    uniq = np.unique(lbl)
    classes.update(uniq.tolist())     # 收集所有出现过的类别

    cls = int(lbl.max())              # 用最大值判定 patch 的“类别”
    counter[cls] += 1

print("出现过的类别：", sorted(classes))  
print("各类别 patch 数量：")
for cls in sorted(counter):
    print(f"  类别 {cls} : {counter[cls]} 个")

#%%
import os
import glob
import random
import shutil
import numpy as np

# 固定随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 原始训练集目录
train_dir = "/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel_v2/data/HYS_test"
# 要存放抽样结果的目录
background_subset_dir = "/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel_v2/data/test"
os.makedirs(background_subset_dir, exist_ok=True)

# 找到所有 .npz 文件
npz_files = glob.glob(os.path.join(train_dir, "*.npz"))

# 1) 拷贝背景样本（lbl 全部为 0），随机抽取不超过 1000 个
background_files = []
for fpath in npz_files:
    lbl = np.load(fpath)["lbl"]
    if int(lbl.max()) == 0:
        background_files.append(fpath)

n_bg = min(100, len(background_files))
selected_bg = random.sample(background_files, n_bg)
for src in selected_bg:
    dst = os.path.join(background_subset_dir, os.path.basename(src))
    shutil.copy(src, dst)
print(f"已将 {n_bg} 个背景样本拷贝到：{background_subset_dir}")

# 2) 拷贝大血管样本（1 类），不做抽样，全部拷贝
vessel1_files = []
for fpath in npz_files:
    lbl = np.load(fpath)["lbl"]
    if int(lbl.max()) == 1:
        vessel1_files.append(fpath)

for src in vessel1_files:
    dst = os.path.join(background_subset_dir, os.path.basename(src))
    shutil.copy(src, dst)
print(f"已将 {len(vessel1_files)} 个大血管（1 类）样本拷贝到：{background_subset_dir}")

# 3) 拷贝小血管样本（2 类），不做抽样，全部拷贝
vessel2_files = []
for fpath in npz_files:
    lbl = np.load(fpath)["lbl"]
    if int(lbl.max()) == 2:
        vessel2_files.append(fpath)

for src in vessel2_files:
    dst = os.path.join(background_subset_dir, os.path.basename(src))
    shutil.copy(src, dst)
print(f"已将 {len(vessel2_files)} 个小血管（2 类）样本拷贝到：{background_subset_dir}")
