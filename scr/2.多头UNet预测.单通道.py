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

#%% 多头模型
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
model_path   = "/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/best_multihead.pth"
czi_path     = "/public2/chengrm/3D_TME/raw_czi/CGL_FFPE_16_L_ELOVL5Gd_CD31_R_COX2Yb_CD66b_062025.czi"
output_dir   = "/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead"
patch_size   = 256
overlap_frac = 0.75
stride       = int(patch_size * (1 - overlap_frac))
disk_radius  = 3

os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# —— 加载模型 —— #
net = MultiHeadUNet(in_ch=1).to(device)
net.load_state_dict(torch.load(model_path, map_location=device))
net.eval()

# —— 形态学结构元 —— #
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (2*disk_radius+1, 2*disk_radius+1)
)

# —— 读取 CZI 并提取 CD31 通道 —— #
img = AICSImage(czi_path)
img.set_scene(0)
raw    = img.get_image_data("CYX", S=0, T=0, Z=0)  # (C,H,W)
cd31   = raw[2].astype(np.float32)
cd31   = cv2.normalize(cd31, None, 0, 1, cv2.NORM_MINMAX)
H, W   = cd31.shape

# —— Pad to fit sliding window —— #
pad_h  = (stride - (H - patch_size) % stride) % stride
pad_w  = (stride - (W - patch_size) % stride) % stride
img_pad = np.pad(cd31, ((0,pad_h),(0,pad_w)), mode="reflect")
H_p, W_p = img_pad.shape

# —— 分别累积大/小血管的投票数 & 投票次数 —— #
sum_large = np.zeros((H_p, W_p), dtype=np.uint16)
sum_small = np.zeros((H_p, W_p), dtype=np.uint16)
cnt       = np.zeros((H_p, W_p), dtype=np.uint16)

with torch.no_grad():
    for y in range(0, H_p-patch_size+1, stride):
        for x in range(0, W_p-patch_size+1, stride):
            patch = img_pad[y:y+patch_size, x:x+patch_size]
            inp   = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,PS,PS]
            o1, o2 = net(inp)                                                   # 两个 head 的 logits: [1,1,PS,PS]
            p1 = torch.sigmoid(o1)[0,0].cpu().numpy()  # 大血管 概率
            p2 = torch.sigmoid(o2)[0,0].cpu().numpy()  # 小血管 概率

            # 二值化阈值可调
            b1 = (p1 > 0.5).astype(np.uint8)
            b2 = (p2 > 0.5).astype(np.uint8)

            sum_large[y:y+patch_size, x:x+patch_size] += b1
            sum_small[y:y+patch_size, x:x+patch_size] += b2
            cnt[y:y+patch_size, x:x+patch_size]       += 1

# —— 投票融合 —— #
# 仅当超过90% 的覆盖都预测为血管时才保留
mask_large = (sum_large / cnt > 0.1).astype(np.uint8) * 255
mask_small = (sum_small / cnt > 0.5).astype(np.uint8) * 255
mask_large = mask_large[:H, :W]
mask_small = mask_small[:H, :W]

# —— 形态学后处理 —— #
# mask_large = cv2.morphologyEx(mask_large, cv2.MORPH_OPEN,  kernel)
# mask_large = cv2.morphologyEx(mask_large, cv2.MORPH_CLOSE, kernel)
# mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN,  kernel)
# mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel)

# —— 保存 —— #
cv2.imwrite(os.path.join(output_dir, "large_vessel.v2.png"), mask_large)
cv2.imwrite(os.path.join(output_dir, "small_vessel.v2.png"), mask_small)

#%% 叠加CD31通道
img = AICSImage(czi_path)
img.set_scene(0)
raw_img    = img.get_image_data("CYX", S=0, T=0, Z=0)  # (C,H,W)
CD31_raw = raw_img[2]

CD31_norm = np.clip((CD31_raw[2].astype(np.float32) / CD31_raw[2].max()) * 3, 0, 1)
CD31_u8 = (CD31_norm * 255).astype(np.uint8)

vis = np.zeros((H, W, 3), dtype=np.uint8)
vis[...,0] = CD31_u8   # 红色通道
vis[...,1] = mask_small * 255   # 绿色通道
vis[...,2] = mask_large * 255
cv2.imwrite(os.path.join(output_dir, "vessel_overlay.png"), vis)

