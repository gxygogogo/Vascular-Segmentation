import os
import cv2
import random
import pickle
import numpy as np
from aicsimageio import AICSImage

#%% 1. 参数设置
patch_size       = 256
neg_discard_rate = 0.95
rotations        = [0, 90, 180, 270]

sample = 'HML_14_FFPE_CD31_555_ELOVL5_647_2025_07_08_2066'
sample_name = 'HML'

# 原图 & 各 mask 路径
img_path             = f'/public5/3D-TME/1.IF/1.raw_czi/{sample}.czi'
vascular_1_mask_path = '/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/HML_model/HML_Mask1.jpg'
vascular_2_mask_path = '/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/HML_model/HML_Mask2.jpg'

# 输出目录
base_out    = f'/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/HML_model/data'
patch_dir   = os.path.join(base_out, f'{sample}_patches')
os.makedirs(patch_dir, exist_ok=True)

#%% 2. 读取原图 + 各 mask
img = AICSImage(img_path)
img.set_scene(0)
raw = img.get_image_data("CYX", S = 0, T = 0, Z = 0)  # (C,H,W)

# 两个通道：CD31 和 DAPI
cd31 = raw[2].astype(np.float32)
dapi = raw[0].astype(np.float32)

# 归一化到 uint8
cd31 = cv2.normalize(cd31, None, 0,255,cv2.NORM_MINMAX).astype(np.uint8)
dapi = cv2.normalize(dapi, None, 0,255,cv2.NORM_MINMAX).astype(np.uint8)

# 二值掩码
m1 = cv2.imread(vascular_1_mask_path, cv2.IMREAD_GRAYSCALE)
m2 = cv2.imread(vascular_2_mask_path, cv2.IMREAD_GRAYSCALE)

H, W = cd31.shape

h1, w1 = m1.shape[:2]
if (h1, w1) != (H, W):
    m1 = cv2.resize(m1, (W, H), interpolation=cv2.INTER_NEAREST)
    m2 = cv2.resize(m2, (W, H), interpolation=cv2.INTER_NEAREST)

# 最终二值化
m1 = (m1 > 127).astype(np.uint8)
m2 = (m2 > 127).astype(np.uint8)

# 检查图像映射是否成功
# alpha_cd31, beta_cd31 = 1.8, 0    # 根据需要调整 alpha>1 提亮
# bright_cd31 = cv2.convertScaleAbs(cd31, alpha=alpha_cd31, beta=beta_cd31)
# canvas = np.zeros((H, W, 3), dtype=np.uint8)
# canvas[..., 0] = (m1 * 255).astype(np.uint8)   # 第一通道：Mask1
# canvas[..., 1] = (m2 * 255).astype(np.uint8)   # 第二通道：Mask2
# canvas[..., 2] = bright_cd31                          # 第三通道：CD31 原始灰度
# cv2.imwrite(os.path.join('/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel', "overlay.png"), canvas)

#%% 3. 切 patch 并保存 .npz，同时记录 patch_info
patch_info = []
idx = 0
n_rows = H // patch_size
n_cols = W // patch_size

for i in range(n_rows):
    for j in range(n_cols):
        y0, x0 = i*patch_size, j*patch_size
        y1, x1 = y0+patch_size, x0+patch_size

        c_p = cd31[y0:y1, x0:x1]
        d_p = dapi[y0:y1, x0:x1]
        m1_p = m1[y0:y1, x0:x1]
        m2_p = m2[y0:y1, x0:x1]
        any_v = bool((m1_p|m2_p).any())

        # 丢弃部分背景
        if not any_v and random.random() < neg_discard_rate:
            idx += 1
            continue

        angles = rotations if any_v else [0]
        for angle in angles:
            M    = cv2.getRotationMatrix2D((patch_size/2, patch_size/2), angle, 1.0)
            c_r  = cv2.warpAffine(c_p,  M, (patch_size, patch_size))
            d_r  = cv2.warpAffine(d_p,  M, (patch_size, patch_size))
            lbl0 = (m1_p*1 + m2_p*2).astype(np.uint8)
            l_r  = cv2.warpAffine(lbl0, M, (patch_size, patch_size), flags = cv2.INTER_NEAREST)

            # 保存为 npz
            fname = f"{sample_name}_{idx:05d}_{angle}.npz"
            np.savez_compressed(
                os.path.join(patch_dir, fname),
                cd31 = c_r,
                dapi = d_r,
                lbl  = l_r
            )

            patch_info.append({
                'idx':       idx,
                'angle':     angle,
                'x0':        x0,
                'y0':        y0,
                'file':      fname,
                'lbl_vals':  np.unique(l_r).tolist(),  # [0,1,2]
                'subset':    None
            })
        idx += 1

print(f"共生成 {idx} 个 patch(含所有类别)。")

#%% 4. 按 class 10% 划分 train/test 并更新 subset
import shutil
groups = {}
for info in patch_info:
    groups.setdefault(info['idx'], []).append(info)

# idx 对应类别 = 该 idx 下任意 patch 的 lbl max
idx2cls = {i: max(g[0]['lbl_vals']) for i,g in groups.items()}

class_to_idxs = {}
for i,cls in idx2cls.items():
    class_to_idxs.setdefault(cls, []).append(i)

random.seed(42)
test_ids = set()
for cls, ids in class_to_idxs.items():
    n = max(1, int(len(ids)*0.1))
    test_ids.update(random.sample(ids, n))

for split in (f'{sample_name}_train', f'{sample_name}_test'):
    os.makedirs(os.path.join(base_out, split), exist_ok=True)

for idx_key, infos in groups.items():
    split = f'{sample_name}_test' if idx_key in test_ids else f'{sample_name}_train'
    for info in infos:
        src = os.path.join(patch_dir, info['file'])
        dst = os.path.join(base_out, split, info['file'])
        shutil.copy(src, dst)
        info['subset'] = split

#%% 5. 保存 patch_info 带子集标记
with open(os.path.join(base_out, f'{sample_name}_patch_info.pkl'), 'wb') as f:
    pickle.dump(patch_info, f)
print("patch_info 已保存，包含 img_array/lbl_array 路径和 subset。")
