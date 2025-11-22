import os
import cv2
import random
import pickle
import numpy as np
from aicsimageio import AICSImage

#%% 1. 参数设置
patch_size       = 256
neg_discard_rate = 0.8
rotations        = [0, 90, 180, 270]

# 原图 & 各 mask 路径
img_path             = '/public2/chengrm/3D_TME/raw_czi/CGL_FFPE_16_L_ELOVL5Gd_CD31_R_COX2Yb_CD66b_062025.czi'
vascular_1_mask_path = '/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/vascular_1.jpg'
vascular_2_mask_path = '/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/vascular_2.jpg'

# 输出目录
base_out     = f'/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/Version_{patch_size}'
out_img_dir  = os.path.join(base_out, 'images')
out_lbl_dir  = os.path.join(base_out, 'labels')
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

#%% 2. 读取原图 + 各 mask
img = AICSImage(img_path); img.set_scene(0)
raw = img.get_image_data("CYX", S=0, T=0, Z=0)  # (C,H,W)

CD31 = raw[2].astype(np.float32)
CD31 = cv2.normalize(CD31, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

m1 = (cv2.imread(vascular_1_mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
m2 = (cv2.imread(vascular_2_mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

H, W = CD31.shape

#%% 3. 切 patch 并保存，同时记录 patch_info
patch_info = []
idx = 0
n_rows = H // patch_size
n_cols = W // patch_size

for i in range(n_rows):
    for j in range(n_cols):
        y0, x0 = i*patch_size, j*patch_size
        y1, x1 = y0+patch_size, x0+patch_size

        img_p   = CD31[y0:y1, x0:x1]
        m1_p    = m1[y0:y1, x0:x1]
        m2_p    = m2[y0:y1, x0:x1]
        any_m   = (m1_p | m2_p).any()

        # 丢弃一部分背景
        if not any_m and random.random() < neg_discard_rate:
            idx += 1
            continue

        angles = rotations if any_m else [0]

        for angle in angles:
            M     = cv2.getRotationMatrix2D((patch_size/2, patch_size/2), angle, 1.0)
            im_r  = cv2.warpAffine(img_p,  M, (patch_size, patch_size))
            lbl0  = (m1_p*1 + m2_p*2).astype(np.uint8)
            lbl_r = cv2.warpAffine(lbl0,   M, (patch_size, patch_size), flags=cv2.INTER_NEAREST)

            # 保存到磁盘
            fname = f"{idx:05d}_{angle}.png"
            cv2.imwrite(os.path.join(out_img_dir, fname), im_r)
            cv2.imwrite(os.path.join(out_lbl_dir, fname), lbl_r)

            # 同时把数组写进 patch_info
            patch_info.append({
                'idx':         idx,
                'angle':       angle,
                'x0':          x0,
                'y0':          y0,
                'img_path':    os.path.join(out_img_dir, fname),
                'lbl_path':    os.path.join(out_lbl_dir, fname),
                'img_array':   im_r.copy(),    # <-- 原始灰度 patch
                'lbl_array':   lbl_r.copy(),   # <-- 对应的 0/1/2 标签矩阵
                'subset':      None
            })
        idx += 1

print(f"共生成 {idx} 个 patch（含所有类别）。")

#%% 4. 按 class 10% 划分 train/test
import shutil

# 按 idx 聚合
groups = {}
for info in patch_info:
    groups.setdefault(info['idx'], []).append(info)

# 先为每个 idx 决定它是哪一类：读第一个 lbl_array 的 max
idx2cls = {idx_key: int(infos[0]['lbl_array'].max()) for idx_key, infos in groups.items()}

# 按 class 收集所有 idx
class_to_idxs = {}
for idx_key, cls in idx2cls.items():
    class_to_idxs.setdefault(cls, []).append(idx_key)

# 每类抽 10% idx 作为测试集
random.seed(42)
test_ids = set()
for cls, ids in class_to_idxs.items():
    n_test = max(1, int(len(ids)*0.1))
    test_ids.update(random.sample(ids, n_test))

# 建目录
for split in ('train','test'):
    os.makedirs(os.path.join(base_out, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_out, split, 'labels'), exist_ok=True)

# 复制文件并写入 subset
for idx_key, infos in groups.items():
    split = 'test' if idx_key in test_ids else 'train'
    for info in infos:
        fn = os.path.basename(info['img_path'])
        shutil.copy(info['img_path'],  os.path.join(base_out, split, 'images', fn))
        shutil.copy(info['lbl_path'],  os.path.join(base_out, split, 'labels', fn))
        info['subset'] = split

#%% 5. 保存包含数组的 patch_info
with open(os.path.join(base_out, 'patch_info.pkl'), 'wb') as f:
    pickle.dump(patch_info, f)

print("所有 patch 信息连同 img_array/lbl_array 已保存到 patch_info.pkl")


lbl = cv2.imread('/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/Version_256/train/labels/04756_90.png', cv2.IMREAD_GRAYSCALE)
print(np.unique(lbl))  # 应该正好得到 array([0,1,2])

