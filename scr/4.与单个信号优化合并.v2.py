import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from aicsimageio import AICSImage

#%%
sample_name = 'CGL_FFPE_16_L_ELOVL5Gd647_CD31_555_R_COX2Yb647_CD66b_555_062025'
czi_path = '/public5/3D-TME/1.IF/1.raw_czi'
save_dir = f'/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}/{sample_name}_enhanced.merge.white.png'
large_mask = f'/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}/step2_connected.png'
small_mask = f'/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}/{sample_name}_small_vessel_pred.png'
cd31_signal = cv2.imread('/public2/chengrm/3D_TME/2.process_data/CGL_FFPE_16_L_ELOVL5Gd_CD31_R_COX2Yb_CD66b_062025_L.CD31/enhanced_CD31_Redsignal.png')
elovl5_signal = cv2.imread('/public2/chengrm/3D_TME/2.process_data/CGL_FFPE_16_L_ELOVL5Gd_CD31_R_COX2Yb_CD66b_062025_L.ELOVL5/enhanced_ELOVL5_Greensignal.png')

shrunk_rate = 11
# 提亮系数
alpha, beta = 2.5, 0  # Bright = orig * 1.5 + 0, 数值上限 255

#%%
# —— 1. 读原始 CD31 图（uint8 BGR） —— #
img = AICSImage(os.path.join(czi_path, sample_name + '.czi'))
img.set_scene(0)
raw = img.get_image_data("CYX", S=0, T=0, Z=0)  # (C,H,W)
cd31 = raw[2].astype(np.float32)
cd31_u8 = cv2.normalize(cd31, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

dapi = raw[2].astype(np.float32)
dapi_u8 = cv2.normalize(dapi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


H, W = cd31_u8.shape

# —— 2. 读掩码 & 生成二值化布尔索引 —— #
# 读取掩码
l_mask = cv2.imread(large_mask, cv2.IMREAD_GRAYSCALE)
s_mask = cv2.imread(small_mask, cv2.IMREAD_GRAYSCALE)

# 合并两个掩码（255 表示要显示的区域）
region_mask = cv2.bitwise_or(l_mask, s_mask)
# 索小掩码
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2 * shrunk_rate + 1, 2 * shrunk_rate + 1))
shrunk_mask = cv2.erode(region_mask, kernel)

dapi_clean = dapi_u8.copy()
dapi_clean[shrunk_mask==255] = 0

cd31_clean = cd31_u8.copy()
cd31_clean[shrunk_mask==255] = 0

elovl5 = elovl5_signal[..., 1].copy()
elovl5[shrunk_mask==255] = 0

cd31 = cd31_signal[..., 2].copy()
cd31[shrunk_mask==255] = 0

# 计算血管壁 & 布尔掩码
orig_mask   = (region_mask == 255)
shrunk_mask = (shrunk_mask == 255)
wall_mask   = orig_mask & ~shrunk_mask   # bool 数组

# 直接用 bool 掩码，不必 threshold：
mask_bin = wall_mask

# 制作提亮版
orig_bgr = cv2.cvtColor(cd31_clean, cv2.COLOR_GRAY2BGR)
bright   = cv2.convertScaleAbs(orig_bgr, alpha=alpha, beta=beta)

# 只在血管壁区域替换
out = orig_bgr.copy()
out[mask_bin] = bright[mask_bin]

# 叠加其它通道，比如 ELOVL5 绿色和 CD31 红色
canvas = np.zeros_like(out)
canvas[...,0] = dapi_clean
canvas[...,1] = elovl5
canvas[...,2] = out[..., 2] + cd31

cv2.imwrite(save_dir, canvas)

## 将组织轮廓外部设置为白色
cv2.imwrite(save_dir, out)
