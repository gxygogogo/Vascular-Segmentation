#%% 导入库
import os
import gc
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from aicsimageio import AICSImage
from torch.utils.data import DataLoader, random_split, Dataset

#%% 模型
class MultiScaleModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1)
        self.branch2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.branch3 = nn.Conv2d(in_channels, out_channels, 3, padding=12,dilation=12)
        self.branch4 = nn.Conv2d(in_channels, out_channels, 3, padding=18,dilation=18)
        self.project = nn.Conv2d(out_channels*4, in_channels, 1)
        self.relu    = nn.ReLU(inplace=True)
    def forward(self, x):
        b1 = self.relu(self.branch1(x))
        b2 = self.relu(self.branch2(x))
        b3 = self.relu(self.branch3(x))
        b4 = self.relu(self.branch4(x))
        out = torch.cat([b1,b2,b3,b4],dim=1)
        return self.relu(self.project(out))

class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(c_out,c_out,3,padding=1),    nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.net(x)

class MultiHeadUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, dropout=0.3):
        super().__init__()
        # --- encoder ---
        self.down1 = DoubleConv(in_ch, base_ch, dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_ch, base_ch*2, dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base_ch*2, base_ch*4, dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base_ch*4, base_ch*8, dropout)
        self.pool4 = nn.MaxPool2d(2)

        # --- bridge + ASPP ---
        self.bridge = DoubleConv(base_ch*8, base_ch*16, dropout)
        self.ms     = MultiScaleModule(base_ch*16, base_ch*4)

        # --- decoder head for large vessels ---
        self.up4_a = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.dec4_a= DoubleConv(base_ch*8*2, base_ch*8, dropout)
        self.up3_a = nn.ConvTranspose2d(base_ch*8, base_ch*4,  2, stride=2)
        self.dec3_a= DoubleConv(base_ch*4*2, base_ch*4, dropout)
        self.up2_a = nn.ConvTranspose2d(base_ch*4, base_ch*2,  2, stride=2)
        self.dec2_a= DoubleConv(base_ch*2*2, base_ch*2, dropout)
        self.up1_a = nn.ConvTranspose2d(base_ch*2, base_ch,    2, stride=2)
        self.dec1_a= DoubleConv(base_ch*2, base_ch, dropout)
        self.out_a = nn.Conv2d(base_ch, 1, 1)  # 输出大血管

        # --- decoder head for small vessels ---
        self.up4_b = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.dec4_b= DoubleConv(base_ch*8*2, base_ch*8, dropout)
        self.up3_b = nn.ConvTranspose2d(base_ch*8, base_ch*4,  2, stride=2)
        self.dec3_b= DoubleConv(base_ch*4*2, base_ch*4, dropout)
        self.up2_b = nn.ConvTranspose2d(base_ch*4, base_ch*2,  2, stride=2)
        self.dec2_b= DoubleConv(base_ch*2*2, base_ch*2, dropout)
        self.up1_b = nn.ConvTranspose2d(base_ch*2, base_ch,    2, stride=2)
        self.dec1_b= DoubleConv(base_ch*2, base_ch, dropout)
        self.out_b = nn.Conv2d(base_ch, 1, 1)  # 输出小血管

    def forward(self,x):
        # encoder
        d1 = self.down1(x); p1=self.pool1(d1)
        d2 = self.down2(p1); p2=self.pool2(d2)
        d3 = self.down3(p2); p3=self.pool3(d3)
        d4 = self.down4(p3); p4=self.pool4(d4)
        # bridge
        b  = self.bridge(p4)
        b  = self.ms(b)
        # head A (large)
        a = self.up4_a(b); a = torch.cat([a,d4],1); a=self.dec4_a(a)
        a = self.up3_a(a); a = torch.cat([a,d3],1); a=self.dec3_a(a)
        a = self.up2_a(a); a = torch.cat([a,d2],1); a=self.dec2_a(a)
        a = self.up1_a(a); a = torch.cat([a,d1],1); a=self.dec1_a(a)
        out_a = self.out_a(a)
        # head B (small)
        b2= self.up4_b(b); b2= torch.cat([b2,d4],1); b2=self.dec4_b(b2)
        b2= self.up3_b(b2);b2= torch.cat([b2,d3],1); b2=self.dec3_b(b2)
        b2= self.up2_b(b2);b2= torch.cat([b2,d2],1); b2=self.dec2_b(b2)
        b2= self.up1_b(b2);b2= torch.cat([b2,d1],1); b2=self.dec1_b(b2)
        out_b = self.out_b(b2)
        return out_a, out_b

#%% 预测
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# CGL_FFPE_16_L_ELOVL5Gd_CD31_R_COX2Yb_CD66b_062025
sample_name  = "CGL_FFPE_16_L_ELOVL5Gd647_CD31_555_R_COX2Yb647_CD66b_555_062025"
czi_path     = "/public5/3D-TME/1.IF/1.raw_czi"
model_path   = "/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel_CGL_v2/best_multihead.pth"
output_dir   = f'/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel_CGL_v2/{sample_name}'
patch_size   = 256
overlap_frac = 0.75
stride       = int(patch_size * (1 - overlap_frac))
disk_radius  = 3

os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# —— 加载模型 —— #
net = MultiHeadUNet(in_ch=2).to(device)
net.load_state_dict(torch.load(model_path, map_location=device))
net.eval()

# —— 形态学结构元 —— #
kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*disk_radius+1,2*disk_radius+1))
kernel_close = kernel_open

# —— 读取 CZI 两个通道 —— #
img = AICSImage(f'{czi_path}/{sample_name}.czi')
img.set_scene(0)
raw = img.get_image_data("CYX", S = 0, T = 0, Z = 0)  # (C, H, W)
# dapi_raw = raw[0].astype(np.float32)
# cd31_raw = raw[1].astype(np.float32)

# 归一化到 [0,1]
dapi = cv2.normalize(raw[0].astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
cd31 = cv2.normalize(raw[1].astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

del img, raw
gc.collect()

# del cd31_raw
# del dapi_raw        
# import gc
# gc.collect()

H, W = dapi.shape

# —— 边界 pad，使得 (Hpad - patch_size) % stride == 0 —— #
pad_h = (stride - (H - patch_size) % stride) % stride
pad_w = (stride - (W - patch_size) % stride) % stride
dapi_pad = np.pad(dapi, ((0,pad_h),(0,pad_w)), mode="reflect")
cd31_pad = np.pad(cd31,((0,pad_h),(0,pad_w)),mode="reflect")
H_p, W_p = dapi_pad.shape

del dapi, cd31
gc.collect()

# —— 累积预测 —— #
sum_large = np.zeros((H_p,W_p),dtype=np.uint16)
sum_small = np.zeros((H_p,W_p),dtype=np.uint16)
cnt       = np.zeros((H_p,W_p),dtype=np.uint16)

print("开始预测")
with torch.no_grad():
    for y in range(0, H_p-patch_size+1, stride):
        for x in range(0, W_p-patch_size+1, stride):
            # 构造双通道 patch
            patch = np.stack([
                cd31_pad[y:y+patch_size, x:x+patch_size],
                dapi_pad[y:y+patch_size, x:x+patch_size]
            ], axis=0)  # [2,H,W]
            inp = torch.from_numpy(patch).unsqueeze(0).to(device)  # [1,2,H,W]

            out1, out2 = net(inp)           # 两个 head 输出 [1,1,H,W]
            p1 = torch.sigmoid(out1)[0,0].cpu().numpy()
            p2 = torch.sigmoid(out2)[0,0].cpu().numpy()

            b1 = (p1>0.5).astype(np.uint8)
            b2 = (p2>0.5).astype(np.uint8)

            sum_large[y:y+patch_size, x:x+patch_size] += b1
            sum_small[y:y+patch_size, x:x+patch_size] += b2
            cnt      [y:y+patch_size, x:x+patch_size] += 1

del dapi_pad, cd31_pad
gc.collect()

print("结束预测")
# —— 投票融合 —— #
# mask_large = ((sum_large/cnt) > 0.4).astype(np.uint8)*255
# mask_small = ((sum_small/cnt) > 0.7).astype(np.uint8)*255
# mask_large = mask_large[:H, :W]
# mask_small = mask_small[:H, :W]

print("形态学处理")
# —— 投票融合 —— #
# 大血管阈值 0.4 === 2/5
# mask_large 中 (sum_large / cnt > 0.4) 等价于 (sum_large * 5 > cnt * 2)
mask_large = np.zeros_like(sum_large, dtype=np.uint8)
mask_large[(sum_large * 5) > (cnt * 2)] = 255

# 小血管阈值 0.7 === 7/10
mask_small = np.zeros_like(sum_small, dtype=np.uint8)
mask_small[(sum_small * 10) > (cnt * 7)] = 255

# 裁回原始大小
mask_large = mask_large[:H, :W]
mask_small = mask_small[:H, :W]

# —— 形态学后处理 —— #
mask_large = cv2.morphologyEx(mask_large, cv2.MORPH_CLOSE, kernel_close)
mask_large = cv2.morphologyEx(mask_large, cv2.MORPH_OPEN,  kernel_open)
mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel_close)
mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN,  kernel_open)

# —— 保存 —— #
cv2.imwrite(os.path.join(output_dir, f'{sample_name}_large_vessel_pred.png'), mask_large)
cv2.imwrite(os.path.join(output_dir, f'{sample_name}_small_vessel_pred.png'), mask_small)

# —— 可视化叠加 —— #
# CD31 原图转 uint8
cd31_u8 = (np.clip(cd31,0,1)*255).astype(np.uint8)
vis = cv2.cvtColor(cd31_u8, cv2.COLOR_GRAY2BGR)
vis[mask_large>0] = (0,0,255)   # 红色表示大血管
vis[mask_small>0] = (0,255,0)   # 绿色表示小血管
cv2.imwrite(os.path.join(output_dir, f'{sample_name}_vessel_overlay.png'), vis)

# —— 形态学后处理 —— #
mask_large = cv2.morphologyEx(mask_large, cv2.MORPH_CLOSE, kernel_close)
mask_large = cv2.morphologyEx(mask_large, cv2.MORPH_OPEN,  kernel_open)
mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel_close)
mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN,  kernel_open)

# —— 保存 —— #
cv2.imwrite(os.path.join(output_dir, f'{sample_name}_large_vessel_pred.png'), mask_large)
cv2.imwrite(os.path.join(output_dir, f'{sample_name}_small_vessel_pred.png'), mask_small)

# —— 可视化叠加 —— #
# CD31 原图转 uint8
cd31_u8 = (np.clip(cd31,0,1)*255).astype(np.uint8)
vis = cv2.cvtColor(cd31_u8, cv2.COLOR_GRAY2BGR)
vis[mask_large>0] = (0,0,255)   # 红色表示大血管
vis[mask_small>0] = (0,255,0)   # 绿色表示小血管
cv2.imwrite(os.path.join(output_dir, f'{sample_name}_vessel_overlay.png'), vis)

