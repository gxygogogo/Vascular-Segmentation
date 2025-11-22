'''
成熟血管掩码处理流程
1. 通过DAPI通道绘制组织轮廓
2. 通过组织轮廓过滤掉轮廓上的假阳性血管掩码
3. 通过连通域分析，进行空洞填充，以及基于阈值过滤噪音
'''
import os
## 设置 OpenCV 读取图片像素的最大约束，需在导入 cv2 之前使用
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(10**10)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from aicsimageio import AICSImage
from PIL import Image
#from skimage import io
Image.MAX_IMAGE_PIXELS = None 

#%% DAPI通道生成组织轮廓
# ——— 参数 ———
sample_name = 'YP_3_13_NORMAL_CD31_555_ELOVL5_647_FFPE_2025_06_30_1861'
img_path    = f'/public5/3D-TME/1.IF/1.raw_czi/{sample_name}.czi'
save_dir    = f'/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}_enh'

# 1. 读 DAPI 通道，归一化到 uint8
img = AICSImage(img_path)
raw = img.get_image_data("CYX", S = 0, T = 0, Z = 0)  # (C, H, W)
# cd31 = raw[2].astype(np.float32)
# cd31_u8 = cv2.normalize(cd31, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# 
# # ——— 2. 二值化 —— Otsu 自动阈值法 —— 
# #    如果你想用固定阈值（比如 50），可以把 cv2.THRESH_OTSU 去掉，第二个参数改成 50。
# _, cd31_bin = cv2.threshold(cd31_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 
# # ——— 3. 可选：小的噪声去除 —— 使用开运算 —— 
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
# cd31_bin = cv2.morphologyEx(cd31_bin, cv2.MORPH_OPEN, kernel, iterations=1)
# 
# # ——— 4. 保存或者后续使用 —— 
# cv2.imwrite(f'{save_dir}/cd31_binary.png', cd31_bin)

image_dapi = img.get_image_data("YX", T=0, C=0, Z=0)
image_dapi = cv2.normalize(image_dapi, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
_, thresh = cv2.threshold(image_dapi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1000, 1000))
dilated = cv2.dilate(thresh, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)
contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (501,501))
# smooth = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
# smooth = cv2.morphologyEx(smooth, cv2.MORPH_OPEN,  kernel, iterations=2)
# contours, _ = cv2.findContours(smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if contours:
    areas = [cv2.contourArea(c) for c in contours]
    max_idx = areas.index(max(areas))
    largest_contour = contours[max_idx]
    
image_copy = np.zeros_like(image_dapi)
cv2.drawContours(image_copy, [largest_contour], -1, color=255, thickness=10)
cv2.imwrite(f'{save_dir}/{sample_name}_tissue_contours.png', image_copy)


#%% 对预测的血管掩码进行连通域及过滤处理
# ——— 参数 ———
min_area            = 3000     # 丢弃面积小于此阈值的小连通域
connectivity        = 4       # 连通域方式：4 或 8
close_size          = 91      # 闭运算结构元尺寸
open_size           = 7       # 开运算结构元尺寸
tissue_contour_path = f"/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}/{sample_name}_tissue_contours.png"    # 只有轮廓线的二值图（255=轮廓，0=背景）
vessel_pred_path    = f"/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}/{sample_name}_large_vessel_pred.png"   # 预测出的二值血管掩码（0/255）
out_filtered        = f"/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}/{sample_name}_large_step1_filtered.png"
out_connected       = f"/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}/{sample_name}_large_step2_connected.png"
out_filled          = f"/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}/{sample_name}_large_step3_filled.png"

# 1. 读取并二值化
# contour = io.imread(tissue_contour_path)
# vpred   = io.imread(vessel_pred_path)
contour = cv2.imread(tissue_contour_path, cv2.IMREAD_GRAYSCALE)
vpred   = cv2.imread(vessel_pred_path,    cv2.IMREAD_GRAYSCALE)
_, contour = cv2.threshold(contour, 127, 255, cv2.THRESH_BINARY)
_, vpred    = cv2.threshold(vpred,    127, 255, cv2.THRESH_BINARY)


# —— 2. 提取最外层轮廓 —— #
# RETR_EXTERNAL 只会返回最外层的轮廓
mask_contours, _ = cv2.findContours(vpred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# —— 3. 生成一个“实心”的外部区域掩码 —— #
h, w = vpred.shape
solid = np.zeros((h, w), dtype=np.uint8)
# 将外轮廓内部全部填成 255
cv2.drawContours(solid, mask_contours, -1, color=255, thickness=cv2.FILLED)


# 2. 从轮廓线生成实心的“组织区” mask
h, w = contour.shape
# 在 contour 上做一次“填充”：
filled = np.zeros((h+2, w+2), np.uint8)
mask_for_flood = contour.copy()
cv2.floodFill(mask_for_flood, filled, (0,0), 255)    # 填充背景
tissue_mask = cv2.bitwise_not(mask_for_flood)        # 取反 → 得到内部区域

# dilate_rad 控制边缘带的宽度（像素）
dilate_rad = 150  
kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilate_rad+1, 2*dilate_rad+1))

# 2a. 先对实心组织 mask 膨胀——> 更宽的组织区域
tissue_dilated = cv2.dilate(tissue_mask, kernel_edge)

# 2b. 再对实心组织 mask 腐蚀——> 稍小的组织区域
tissue_eroded  = cv2.erode(tissue_mask, kernel_edge)

# 2c. 用膨胀后减去腐蚀后，得到一条宽度约为 2*dilate_rad 的边缘带
edge_band = cv2.subtract(tissue_dilated, tissue_eroded)
#    edge_band 上的像素 = 255 的位置就是我们要去除的“可疑边缘假阳性”

# ——— 修改第3步，只保留组织内部 *且* 不在边缘带上的预测 ——  
# 原来的：
# inside = cv2.bitwise_and(vpred, tissue_mask)
# 改为：
tmp = cv2.bitwise_and(solid, tissue_mask)                    # 仍然先限制在组织内部
inside = cv2.bitwise_and(tmp, cv2.bitwise_not(edge_band))    # 再去掉边缘带内的所有像素

# 4. 连通域分析，过滤小块
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inside, connectivity=connectivity)
filtered = np.zeros_like(inside)
for lbl in range(1, num_labels):
    area = stats[lbl, cv2.CC_STAT_AREA]
    if area >= min_area:
        filtered[labels == lbl] = 255
cv2.imwrite(out_filtered, filtered)
print("→ 步骤1+2结果保存在: ", out_filtered)

# 5. 形态学闭运算填补断裂
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel_close)

# 6. （可选）开运算去噪并恢复线宽
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
connected = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

cv2.imwrite(out_connected, connected)
print("→ 步骤3结果保存在: ", out_connected)

# 7. 填充每个连通域内部的空洞
#    原理：先对当前二值图取反，然后从图像边缘 floodFill，得到“外部背景”；再取反，得到所有“洞”区域；最后将洞区域和原 mask 做 or。
inv = cv2.bitwise_not(connected)
h, w = inv.shape
floodfill_mask = np.zeros((h+2, w+2), np.uint8)  # floodFill 要求比图大 2px
# 从 (0,0) 开始填充“外部”背景
cv2.floodFill(inv, floodfill_mask, (0, 0), 255)
holes = cv2.bitwise_not(inv)            # 这就是所有的“空洞”
filled = cv2.bitwise_or(connected, holes)

cv2.imwrite(out_filled, filled)
print("→ 步骤4(洞填充)结果保存在: ", out_filled)
 