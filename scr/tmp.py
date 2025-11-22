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
sample_name = 'HLX_22_NORMAL_CD31_555_ELOVL5_647_FFPE_2025_06_30__1859'
img_path    = f'/public5/3D-TME/1.IF/1.raw_czi/{sample_name}.czi'
save_dir    = f'/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}_enh'

min_area            = 3000     # 丢弃面积小于此阈值的小连通域
connectivity        = 4       # 连通域方式：4 或 8
close_size          = 91      # 闭运算结构元尺寸
open_size           = 7       # 开运算结构元尺寸
tissue_contour_path = f"/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}_enh/{sample_name}_tissue_contours.png"    # 只有轮廓线的二值图（255=轮廓，0=背景）
vessel_pred_path    = f"/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}_enh/{sample_name}_large_vessel_pred.png"   # 预测出的二值血管掩码（0/255）
out_filtered        = f"/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}_enh/{sample_name}_large_step1_filtered.png"
out_connected       = f"/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}_enh/{sample_name}_large_step2_connected.png"
out_filled          = f"/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/{sample_name}_enh/{sample_name}_large_step3_filled.png"

# 1. 读取并二值化
# contour = io.imread(tissue_contour_path)
# vpred   = io.imread(vessel_pred_path)

contour = cv2.imread(tissue_contour_path, cv2.IMREAD_GRAYSCALE)
print('已读入组织轮廓')

vpred   = cv2.imread(vessel_pred_path,    cv2.IMREAD_GRAYSCALE)
_, contour = cv2.threshold(contour, 127, 255, cv2.THRESH_BINARY)
_, vpred    = cv2.threshold(vpred,    127, 255, cv2.THRESH_BINARY)
print('已读入血管掩码')

# —— 2. 提取最外层轮廓 —— #
# RETR_EXTERNAL 只会返回最外层的轮廓
mask_contours, _ = cv2.findContours(vpred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# —— 3. 生成一个“实心”的外部区域掩码 —— #
h, w = vpred.shape
solid = np.zeros((h, w), dtype=np.uint8)
# 将外轮廓内部全部填成 255
cv2.drawContours(solid, mask_contours, -1, color=255, thickness=cv2.FILLED)
print('drawContours完成')

# 2. 从轮廓线生成实心的“组织区” mask
h, w = contour.shape
# 在 contour 上做一次“填充”：
filled = np.zeros((h+2, w+2), np.uint8)
mask_for_flood = contour.copy()
cv2.floodFill(mask_for_flood, filled, (0,0), 255)    # 填充背景
tissue_mask = cv2.bitwise_not(mask_for_flood)        # 取反 → 得到内部区域
print('floodFill完成')

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
print('血管轮廓提取完成')

# ——— 修改第3步，只保留组织内部 *且* 不在边缘带上的预测 ——  
# 原来的：
# inside = cv2.bitwise_and(vpred, tissue_mask)
# 改为：
tmp = cv2.bitwise_and(solid, tissue_mask)                    # 仍然先限制在组织内部
inside = cv2.bitwise_and(tmp, cv2.bitwise_not(edge_band))    # 再去掉边缘带内的所有像素

print('进行保存')
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
