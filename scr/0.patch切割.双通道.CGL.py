import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(10**10)
import cv2
import glob
import random
import pickle
import shutil
import numpy as np
from aicsimageio import AICSImage

#%% 1. 参数设置
patch_size = 256
rotations  = [0, 90, 180, 270]

sample      = 'CGL_FFPE_16_L_ELOVL5Gd647_CD31_555_R_COX2Yb647_CD66b_555_062025'
sample_name = 'CGL'

# 原图 & 各 mask 路径
img_path             = f'/public5/3D-TME/1.IF/1.raw_czi/{sample}.czi'
vascular_1_mask_path = '/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel_CGL/vascular_1.jpg'
vascular_2_mask_path = '/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel_CGL/vascular_2.jpg'

# 输出目录
base_out  = f'/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel_CGL/data'
patch_dir = os.path.join(base_out, f'{sample_name}_patches')
os.makedirs(patch_dir, exist_ok=True)

#%% 2. 读取原图 + 各 mask
img = AICSImage(img_path)
img.set_scene(0)
raw = img.get_image_data("CYX", S=0, T=0, Z=0)  # (C, H, W)

cd31 = raw[2].astype(np.float32)
dapi = raw[0].astype(np.float32)
cd31 = cv2.normalize(cd31, None, 0,255,cv2.NORM_MINMAX).astype(np.uint8)
dapi = cv2.normalize(dapi, None, 0,255,cv2.NORM_MINMAX).astype(np.uint8)

m1 = cv2.imread(vascular_1_mask_path, cv2.IMREAD_GRAYSCALE)
m2 = cv2.imread(vascular_2_mask_path, cv2.IMREAD_GRAYSCALE)
H, W = cd31.shape
if m1.shape[:2] != (H, W):
    m1 = cv2.resize(m1, (W, H), interpolation=cv2.INTER_NEAREST)
    m2 = cv2.resize(m2, (W, H), interpolation=cv2.INTER_NEAREST)
m1 = (m1 > 127).astype(np.uint8)
m2 = (m2 > 127).astype(np.uint8)

#%% 3. 切 patch 并保存 .npz，同时记录 patch_info（保留所有负样本）
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

        angles = rotations if any_v else [0]
        for angle in angles:
            M   = cv2.getRotationMatrix2D((patch_size/2, patch_size/2), angle, 1.0)
            c_r = cv2.warpAffine(c_p, M, (patch_size, patch_size))
            d_r = cv2.warpAffine(d_p, M, (patch_size, patch_size))
            lbl0 = (m1_p*1 + m2_p*2).astype(np.uint8)
            l_r  = cv2.warpAffine(lbl0, M, (patch_size, patch_size), flags=cv2.INTER_NEAREST)

            fname = f"{sample_name}_{idx:05d}_{angle}.npz"
            np.savez_compressed(
                os.path.join(patch_dir, fname),
                cd31 = c_r,
                dapi = d_r,
                lbl  = l_r
            )

            patch_info.append({
                'idx':      idx,
                'angle':    angle,
                'x0':       x0,
                'y0':       y0,
                'file':     fname,
                'lbl_vals': np.unique(l_r).tolist(),  # [0] or [0,1,2]
                'neg_ratio': None,
                'neg_group': None
            })
        idx += 1

print(f"共生成 {idx} 个 patch（含所有类别）。")

#%% 4. 创建各分类目录
all_dirs = {
    # 负样本按信号比例
    'neg_0':os.path.join(base_out,'neg_0'),
    'neg_1':os.path.join(base_out,'neg_1'),
    'neg_2':os.path.join(base_out,'neg_2'),
    # 正样本按血管类型
    'pos_1':os.path.join(base_out,'pos_1'),
    'pos_2':os.path.join(base_out,'pos_2'),
}
for d in all_dirs.values(): os.makedirs(d, exist_ok=True)

#%% 5. 负样本分桶 & 正样本分目录
for info in patch_info:
    npz = os.path.join(patch_dir, info['file'])
    data= np.load(npz)
    lbl = data['lbl']
    # Positive if any vessel
    if (lbl>0).any():
        # 按 max(label) 决定放 pos_1 还是 pos_2
        cls = int(lbl.max())
        tgt = all_dirs[f'pos_{cls}']
        info['pos_group'] = f'pos_{cls}'
        shutil.copy(npz, tgt)
    else:
        # 负样本，按底图信号比例分桶
        cd, da = data['cd31'], data['dapi']
        sig = ((cd>0)|(da>0)).sum()
        ratio = float(sig) / (patch_size*patch_size)
        info['neg_ratio']=ratio
        if ratio<0.25:  grp='neg_0'
        elif ratio<0.75:grp='neg_1'
        else:           grp='neg_2'
        info['neg_group']=grp
        shutil.copy(npz, all_dirs[grp])

print("正负样本已分类完成。")

#%% 6. 在每个 neg_x 下随机抽 10 个 patch，可视化 overlay
for d in ('neg_0', 'neg_1', 'neg_2', 'pos_1', 'pos_2'):
    viz_dir = os.path.join(d, 'visuals')
    os.makedirs(viz_dir, exist_ok=True)
    files = glob.glob(os.path.join(d, '*.npz'))
    samples = random.sample(files, min(10, len(files)))

    for p in samples:
        data = np.load(p)
        cd = data['cd31'].astype(np.uint8)
        da = data['dapi'].astype(np.uint8)
        lbl0 = data['lbl'].astype(np.uint8)

        # 底图 BGR: B<-DAPI, G<-CD31
        base = cv2.merge([da, np.zeros_like(cd), cd])

        fn = os.path.basename(p).replace('.npz', '.png')
        cv2.imwrite(os.path.join(viz_dir, fn), base)

print("各目录下的 10 张示例已保存到 visuals 子目录。")

#%% 7. 保存 patch_info（含 neg_ratio & neg_group）
with open(os.path.join(base_out, f'{sample_name}_patch_info.pkl'), 'wb') as f:
    pickle.dump(patch_info, f)
print("完整的 patch_info 已保存。")

#%% 8. 训练集
import os
import glob
import random
import shutil

# 1. 配置参数
src_dir     = '/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel_CGL/data/neg_2'      # 源目录
dst_dir     = '/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel_CGL/data/train'       # 目标 train 目录
pattern     = '*.npz'                    # 要匹配的文件类型
num_samples = 1700                         # 随机抽取的文件数

# 2. 列出所有文件
all_files = glob.glob(os.path.join(src_dir, pattern))
file_nums = len(all_files)
if num_samples > len(all_files):
    for file in all_files:
        shutil.copy(file, dst_dir)

# 3. 随机采样
random.seed(42)  # 可选，保证可复现
samples = random.sample(all_files, num_samples)

# 4. 创建目标目录
os.makedirs(dst_dir, exist_ok=True)

# 5. 复制文件
for src_path in samples:
    filename = os.path.basename(src_path)
    dst_path  = os.path.join(dst_dir, filename)
    shutil.copy(src_path, dst_path)

print(f"已从 {src_dir} 随机复制 {num_samples} 个文件到 {dst_dir}")

