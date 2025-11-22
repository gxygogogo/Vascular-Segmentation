import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(10**10)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from aicsimageio import AICSImage

#%%
sample_names = ['HYS_2_13_NORMAL_CD31_555_ELOVL5_647_FFPE_2025_06_30__1857',
                'GXX_2_20_FFPE_2025_05_09__CD31_594_ELOVL5_488_r0942',
                'GXX_2_21_FFPE_2025_05_09__CD34_488_ELOVL5__594_r0915',
                'YP_3_13_NORMAL_CD31_555_ELOVL5_647_FFPE_2025_06_30_1861',
                'LSN_4_22_FFPE_2025_05_09__CD31_594_ELOVL5_488_CD34_488_ELOVL5_594_0914',
                'HLX_22_NORMAL_CD31_555_ELOVL5_647_FFPE_2025_06_30__1859']
for sample_name in sample_names:
    print(sample_name)
    czi_path = '/public5/3D-TME/1.IF/1.raw_czi'
    save_dir = f'/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}_enh/{sample_name}_enhanced.vessel.v2.png'
    large_mask = f'/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}_enh/{sample_name}_large_step2_connected.png'
    small_mask = f'/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}_enh/{sample_name}_small_vessel_pred.png'
    contour_path = f'/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}_enh/{sample_name}_tissue_contours.png'

    shrunk_rate = 11
    # 提亮系数
    alpha, beta = 5.5, 4.5  # Bright = orig * 1.5 + 0, 数值上限 255
    alpha_global, beta_global = 2.5, 0   # 亮度放大 1.5 倍
    #%%
    # —— 1. 读原始 CD31 图（uint8 BGR） —— #
    img = AICSImage(os.path.join(czi_path, sample_name + '.czi'))
    img.set_scene(0)
    raw = img.get_image_data("CYX", S=0, T=0, Z=0)  # (C,H,W)
    cd31 = raw[2].astype(np.float32)
    cd31_u8 = cv2.normalize(cd31, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    dapi = raw[0].astype(np.float32)
    dapi_u8 = cv2.normalize(dapi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # —— 读取组织轮廓 ——
    contour = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
    cnts, _ = cv2.findContours(contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tissue_solid = np.zeros_like(contour)
    cv2.drawContours(tissue_solid, cnts, -1, 255, cv2.FILLED)

    # —— 去除组织外 ——
    cd31_mask = cd31_u8.copy()
    cd31_mask[tissue_solid == 0] = 0

    dapi_mask = dapi_u8.copy()
    dapi_mask[tissue_solid == 0] = 0

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

    dapi_clean = dapi_mask.copy()
    dapi_clean[shrunk_mask==255] = 0

    cd31_clean = cd31_mask.copy()
    cd31_clean[shrunk_mask==255] = 0

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

    # 叠加DAPI，ELOVL5信号，CD31信号&血管
    canvas = np.zeros_like(out)
    #canvas[...,0] = dapi_clean
    canvas[...,2] = out[..., 2]

    canvas_bright = cv2.convertScaleAbs(canvas, alpha=alpha_global, beta=beta_global)

    cv2.imwrite(save_dir, canvas_bright)


    #%%
    vessel_bin = (region_mask > 128).astype(np.uint8) * 255

    # 找轮廓
    cnts, _ = cv2.findContours(vessel_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 3) 生成彩色 overlay
    H, W = cd31_clean.shape
    border_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(border_mask, cnts, -1, color=1, thickness=1)

    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    overlay[..., 0] = dapi_clean   # B <- DAPI
    overlay[..., 2] = cd31_clean   # G <- CD31
    # R 通道先填 0，后面用 drawContours 画红线

    result = overlay.copy()
    result[border_mask == 1] = (152, 251, 152)

    cv2.imwrite(f'/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}_enh/{sample_name}_enhanced.vessel.overlay.png', result)
