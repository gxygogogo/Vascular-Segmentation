#%% 导入库
import os
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

# —— 参数 ——  
czi_path    = "/public5/3D-TME/1.IF/1.raw_czi/YP_1_14_CD31_555_ELOVL5_647_FFPE_2025_06_30_1860.czi"
model_path  = "/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/best_multihead.pth"
output_dir  = "/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel/YP_1_14_CD31_555_ELOVL5_647_FFPE_2025_06_30_1860"
patch_size  = 256
overlap_frac= 0.75
stride      = int(patch_size * (1 - overlap_frac))
disk_radius = 3

os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
net = MultiHeadUNet(in_ch=2).to(device)
net.load_state_dict(torch.load(model_path, map_location=device))
net.eval()

# Morphology kernels
kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*disk_radius+1,)*2)
kernel_close = kernel_open

# Open CZI lazily
img = AICSImage(czi_path, reconstruct_mosaic=False)
da_cube = img.xarray_dask_data.data  # dims like ("C","Y","X")
cd31 = da_cube.isel(C=2)
dapi = da_cube.isel(C=0)
H, W = dapi.shape

# Lazy pad
pad_h = (stride - (H - patch) % stride) % stride
pad_w = (stride - (W - patch) % stride) % stride
cd31 = da.pad(cd31, ((0,pad_h),(0,pad_w)), mode='reflect')
dapi = da.pad(dapi, ((0,pad_h),(0,pad_w)), mode='reflect')
H_p, W_p = H+pad_h, W+pad_w

# Voting accumulators
sum_L = np.zeros((H_p, W_p), dtype=np.uint16)
sum_S = np.zeros((H_p, W_p), dtype=np.uint16)
cnt   = np.zeros((H_p, W_p), dtype=np.uint16)

with torch.no_grad():
    for y in range(0, H_p-patch+1, stride):
        for x in range(0, W_p-patch+1, stride):
            c_patch = cd31[y:y+patch, x:x+patch].compute().astype(np.float32)
            d_patch = dapi[y:y+patch, x:x+patch].compute().astype(np.float32)
            # normalize
            c_patch = cv2.normalize(c_patch, None, 0,1, cv2.NORM_MINMAX)
            d_patch = cv2.normalize(d_patch, None, 0,1, cv2.NORM_MINMAX)
            inp = torch.from_numpy(np.stack([c_patch, d_patch])).unsqueeze(0).to(device)
            with autocast():
                oL, oS = net(inp)
            pL = (torch.sigmoid(oL)>0.5)[0,0].cpu().numpy().astype(np.uint8)
            pS = (torch.sigmoid(oS)>0.5)[0,0].cpu().numpy().astype(np.uint8)
            sum_L[y:y+patch, x:x+patch] += pL
            sum_S[y:y+patch, x:x+patch] += pS
            cnt  [y:y+patch, x:x+patch] += 1

# Integer thresholding
mask_L = np.zeros_like(sum_L, dtype=np.uint8)
mask_L[(sum_L*5) > (cnt*2)] = 255
mask_S = np.zeros_like(sum_S, dtype=np.uint8)
mask_S[(sum_S*10) > (cnt*7)] = 255

# Crop and post-process
mask_L = mask_L[:H, :W]
mask_S = mask_S[:H, :W]
mask_L = cv2.morphologyEx(mask_L, cv2.MORPH_CLOSE, kernel_close)
mask_L = cv2.morphologyEx(mask_L, cv2.MORPH_OPEN,  kernel_open)
mask_S = cv2.morphologyEx(mask_S, cv2.MORPH_CLOSE, kernel_close)
mask_S = cv2.morphologyEx(mask_S, cv2.MORPH_OPEN,  kernel_open)

cv2.imwrite(f"{output_dir}/{sample}_large_vessel_pred.png", mask_L)
cv2.imwrite(f"{output_dir}/{sample}_small_vessel_pred.png", mask_S)
