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

#%% 双头预测模型
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=True)

    def forward(self, x):
        # Global average pooling
        w = x.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class AttentionGate(nn.Module):
    """
    Attention Gate for skip connections: uses gating signal g and feature map x.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, bias=False)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.last_attention = None

    def forward(self, g, x):
        # g: gating signal (from decoder), x: skip connection (from encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        self.last_attention = psi.detach().cpu().squeeze(0).squeeze(0)
        return x * psi


class DoubleConvSE(nn.Module):
    """
    Two consecutive conv layers + SE attention
    """
    def __init__(self, c_in, c_out, dropout=0.3, reduction=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(c_out, reduction)

    def forward(self, x):
        x = self.net(x)
        return self.se(x)


class MultiScaleModule(nn.Module):
    """
    ASPP-like multi-scale module
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1)
        self.branch2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.branch3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.branch4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.project = nn.Conv2d(out_channels * 4, in_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.relu(self.branch1(x))
        b2 = self.relu(self.branch2(x))
        b3 = self.relu(self.branch3(x))
        b4 = self.relu(self.branch4(x))
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.relu(self.project(out))


class MultiHeadUNetWithAttn(nn.Module):
    """
    Multi-head U-Net with SE channel attention and Attention Gates on skip-connections.
    """
    def __init__(self, in_ch=2, base_ch=64, dropout=0.3, se_reduction=16):
        super().__init__()
        # Encoder with SE blocks
        self.down1 = DoubleConvSE(in_ch, base_ch, dropout, se_reduction)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConvSE(base_ch, base_ch*2, dropout, se_reduction)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConvSE(base_ch*2, base_ch*4, dropout, se_reduction)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConvSE(base_ch*4, base_ch*8, dropout, se_reduction)
        self.pool4 = nn.MaxPool2d(2)

        # Bridge + ASPP
        self.bridge = DoubleConvSE(base_ch*8, base_ch*16, dropout, se_reduction)
        self.ms = MultiScaleModule(base_ch*16, base_ch*4)

        # Attention gates for head A (large vessels)
        self.att4_a = AttentionGate(F_g=base_ch*8, F_l=base_ch*8, F_int=base_ch*4)
        self.att3_a = AttentionGate(F_g=base_ch*4, F_l=base_ch*4, F_int=base_ch*2)
        self.att2_a = AttentionGate(F_g=base_ch*2, F_l=base_ch*2, F_int=base_ch)
        self.att1_a = AttentionGate(F_g=base_ch,   F_l=base_ch,   F_int=base_ch//2)

        # Decoder head for large vessels
        self.up4_a = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.dec4_a = DoubleConvSE(base_ch*8*2, base_ch*8, dropout, se_reduction)
        self.up3_a = nn.ConvTranspose2d(base_ch*8,  base_ch*4, 2, stride=2)
        self.dec3_a = DoubleConvSE(base_ch*4*2, base_ch*4, dropout, se_reduction)
        self.up2_a = nn.ConvTranspose2d(base_ch*4,  base_ch*2, 2, stride=2)
        self.dec2_a = DoubleConvSE(base_ch*2*2, base_ch*2, dropout, se_reduction)
        self.up1_a = nn.ConvTranspose2d(base_ch*2,  base_ch,   2, stride=2)
        self.dec1_a = DoubleConvSE(base_ch*2,   base_ch, dropout, se_reduction)
        self.out_a  = nn.Conv2d(base_ch, 1, 1)

        # Decoder head for small vessels (reuse attention gates or create new ones)
        self.up4_b = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.dec4_b = DoubleConvSE(base_ch*8*2, base_ch*8, dropout, se_reduction)
        self.up3_b = nn.ConvTranspose2d(base_ch*8,  base_ch*4, 2, stride=2)
        self.dec3_b = DoubleConvSE(base_ch*4*2, base_ch*4, dropout, se_reduction)
        self.up2_b = nn.ConvTranspose2d(base_ch*4,  base_ch*2, 2, stride=2)
        self.dec2_b = DoubleConvSE(base_ch*2*2, base_ch*2, dropout, se_reduction)
        self.up1_b = nn.ConvTranspose2d(base_ch*2,  base_ch,   2, stride=2)
        self.dec1_b = DoubleConvSE(base_ch*2,   base_ch, dropout, se_reduction)
        self.out_b  = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x); p1 = self.pool1(d1)
        d2 = self.down2(p1); p2 = self.pool2(d2)
        d3 = self.down3(p2); p3 = self.pool3(d3)
        d4 = self.down4(p3); p4 = self.pool4(d4)

        # Bridge
        b = self.bridge(p4)
        b = self.ms(b)

        # Head A (large vessels)
        a4 = self.up4_a(b)
        d4a = self.att4_a(g=a4, x=d4)
        a = torch.cat([a4, d4a], dim=1); a = self.dec4_a(a)

        a3 = self.up3_a(a)
        d3a = self.att3_a(g=a3, x=d3)
        a = torch.cat([a3, d3a], dim=1); a = self.dec3_a(a)

        a2 = self.up2_a(a)
        d2a = self.att2_a(g=a2, x=d2)
        a = torch.cat([a2, d2a], dim=1); a = self.dec2_a(a)

        a1 = self.up1_a(a)
        d1a = self.att1_a(g=a1, x=d1)
        a = torch.cat([a1, d1a], dim=1); a = self.dec1_a(a)

        out_a = self.out_a(a)

        # Head B (small vessels) - reuse the same gates or new ones
        b4 = self.up4_b(b)
        d4b = self.att4_a(g=b4, x=d4)
        b_ = torch.cat([b4, d4b], dim=1); b_ = self.dec4_b(b_)

        b3 = self.up3_b(b_)
        d3b = self.att3_a(g=b3, x=d3)
        b_ = torch.cat([b3, d3b], dim=1); b_ = self.dec3_b(b_)

        b2 = self.up2_b(b_)
        d2b = self.att2_a(g=b2, x=d2)
        b_ = torch.cat([b2, d2b], dim=1); b_ = self.dec2_b(b_)

        b1 = self.up1_b(b_)
        d1b = self.att1_a(g=b1, x=d1)
        b_ = torch.cat([b1, d1b], dim=1); b_ = self.dec1_b(b_)

        out_b = self.out_b(b_)
        return out_a, out_b


#%% 预测
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# CGL_FFPE_16_L_ELOVL5Gd_CD31_R_COX2Yb_CD66b_062025
sample_name  = "LPN_2_1_CD31_555_ELOVL5_647_DESI_FROZEN_2025_07_02_1981"
czi_path     = "/public5/3D-TME/1.IF/1.raw_czi"
model_path   = "/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel_v3/best_multihead.attention.pth"
output_dir   = f'/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel_v3/{sample_name}'
patch_size   = 256
overlap_frac = 0.75
stride       = int(patch_size * (1 - overlap_frac))
disk_radius  = 3

os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# —— 加载模型 —— #
net = MultiHeadUNetWithAttn(in_ch=2).to(device)
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
att_acc   = np.zeros((H_p,W_p),dtype=np.float32)
att_cnt   = np.zeros((H_p,W_p),dtype=np.uint16)

with torch.no_grad():
    for y in range(0, H_p-patch_size+1, stride):
        for x in range(0, W_p-patch_size+1, stride):
            patch = np.stack([
                cd31_pad[y:y+patch_size, x:x+patch_size],
                dapi_pad[y:y+patch_size, x:x+patch_size]
            ], axis=0)[None]  # [1,2,H,W]
            inp = torch.from_numpy(patch).to(device)

            out1, out2 = net(inp)  # [1,1,H,W] ×2
            p1 = torch.sigmoid(out1)[0,0].cpu().numpy()
            p2 = torch.sigmoid(out2)[0,0].cpu().numpy()

            b1 = (p1>0.5).astype(np.uint8)
            b2 = (p2>0.5).astype(np.uint8)

            sum_large[y:y+patch_size, x:x+patch_size] += b1
            sum_small[y:y+patch_size, x:x+patch_size] += b2
            cnt      [y:y+patch_size, x:x+patch_size] += 1

            # —— attention gate 取自最底层 att1_a.last_attention —— #
            att_patch = net.att1_a.last_attention.cpu().numpy()
            att_acc[y:y+patch_size, x:x+patch_size] += att_patch
            att_cnt[y:y+patch_size, x:x+patch_size] += 1

del dapi_pad, cd31_pad
gc.collect()

# —— 投票融合 —— #
# mask_large = ((sum_large/cnt) > 0.4).astype(np.uint8)*255
# mask_small = ((sum_small/cnt) > 0.7).astype(np.uint8)*255
# mask_large = mask_large[:H, :W]
# mask_small = mask_small[:H, :W]

# —— 投票融合 —— #
# 大血管阈值 0.4 -> 2/5
mask_large = ((sum_large * 5) > (cnt * 2)).astype(np.uint8)*255
# 小血管阈值 0.7 -> 7/10
mask_small = ((sum_small * 10) > (cnt * 7)).astype(np.uint8)*255

mask_large = mask_large[:H, :W]
mask_small = mask_small[:H, :W]

mask_large = cv2.morphologyEx(mask_large, cv2.MORPH_CLOSE, kernel_close)
mask_large = cv2.morphologyEx(mask_large, cv2.MORPH_OPEN,  kernel_open)
mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel_close)
mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN,  kernel_open)

cv2.imwrite(f"{output_dir}/{sample_name}_large_pred.png", mask_large)
cv2.imwrite(f"{output_dir}/{sample_name}_small_pred.png", mask_small)

# attention 热图绘制
# 防止除 0
att_map = att_acc / np.maximum(att_cnt, 1)
att_map = att_map[:H, :W]

plt.figure(figsize=(8,8))
plt.imshow(att_map, cmap='hot', interpolation='bilinear')
plt.axis('off')
plt.colorbar()
plt.savefig(f"{output_dir}/{sample_name}_global_attention.png", dpi=200)
plt.close()

#%%
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

