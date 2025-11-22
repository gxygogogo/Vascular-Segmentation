#%% 导入库
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(10**10)
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split, Dataset

#%% 双头预测模型
class MultiScaleModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, 1, padding = 0, dilation = 1)
        self.branch2 = nn.Conv2d(in_channels, out_channels, 3, padding = 6, dilation = 6)
        self.branch3 = nn.Conv2d(in_channels, out_channels, 3, padding = 12, dilation = 12)
        self.branch4 = nn.Conv2d(in_channels, out_channels, 3, padding = 18, dilation = 18)
        self.project = nn.Conv2d(out_channels*4, in_channels, 1)
        self.relu    = nn.ReLU(inplace=True)
    def forward(self, x):
        b1 = self.relu(self.branch1(x))
        b2 = self.relu(self.branch2(x))
        b3 = self.relu(self.branch3(x))
        b4 = self.relu(self.branch4(x))
        out = torch.cat([b1,b2,b3,b4], dim = 1)
        return self.relu(self.project(out))

class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out, dropout = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding = 1), nn.ReLU(inplace = True),
            nn.Dropout2d(dropout),
            nn.Conv2d(c_out,c_out, 3, padding = 1), nn.ReLU(inplace = True),
        )
    def forward(self, x): 
        return self.net(x)

class MultiHeadUNet(nn.Module):
    def __init__(self, in_ch = 1, base_ch = 64, dropout = 0.3):
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
        self.up4_a = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride = 2)
        self.dec4_a = DoubleConv(base_ch*8*2, base_ch*8, dropout)
        self.up3_a = nn.ConvTranspose2d(base_ch*8, base_ch*4,  2, stride = 2)
        self.dec3_a = DoubleConv(base_ch*4*2, base_ch*4, dropout)
        self.up2_a = nn.ConvTranspose2d(base_ch*4, base_ch*2,  2, stride = 2)
        self.dec2_a = DoubleConv(base_ch*2*2, base_ch*2, dropout)
        self.up1_a = nn.ConvTranspose2d(base_ch*2, base_ch,    2, stride = 2)
        self.dec1_a = DoubleConv(base_ch*2, base_ch, dropout)
        self.out_a = nn.Conv2d(base_ch, 1, 1)  # 输出大血管

        # --- decoder head for small vessels ---
        self.up4_b = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride = 2)
        self.dec4_b = DoubleConv(base_ch*8*2, base_ch*8, dropout)
        self.up3_b = nn.ConvTranspose2d(base_ch*8, base_ch*4,  2, stride = 2)
        self.dec3_b = DoubleConv(base_ch*4*2, base_ch*4, dropout)
        self.up2_b = nn.ConvTranspose2d(base_ch*4, base_ch*2,  2, stride = 2)
        self.dec2_b = DoubleConv(base_ch*2*2, base_ch*2, dropout)
        self.up1_b = nn.ConvTranspose2d(base_ch*2, base_ch,    2, stride = 2)
        self.dec1_b = DoubleConv(base_ch*2, base_ch, dropout)
        self.out_b = nn.Conv2d(base_ch, 1, 1)  # 输出小血管

    def forward(self,x):
        # encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)
        # bridge
        b = self.bridge(p4)
        b = self.ms(b)
        # head A (large)
        a = self.up4_a(b); a = torch.cat([a,d4],1); a=self.dec4_a(a)
        a = self.up3_a(a); a = torch.cat([a,d3],1); a=self.dec3_a(a)
        a = self.up2_a(a); a = torch.cat([a,d2],1); a=self.dec2_a(a)
        a = self.up1_a(a); a = torch.cat([a,d1],1); a=self.dec1_a(a)
        out_a = self.out_a(a)
        # head B (small)
        b2 = self.up4_b(b)
        b2 = torch.cat([b2,d4],1)
        b2 = self.dec4_b(b2)
        b2 = self.up3_b(b2)
        b2 = torch.cat([b2,d3],1)
        b2 = self.dec3_b(b2)
        b2 = self.up2_b(b2)
        b2 = torch.cat([b2,d2],1)
        b2 = self.dec2_b(b2)
        b2 = self.up1_b(b2)
        b2 = torch.cat([b2,d1],1)
        b2 = self.dec1_b(b2)
        out_b = self.out_b(b2)
        return out_a, out_b

#%% 导入单通道数据集
class TwoHeadDataset(Dataset):
    def __init__(self, root, img_folder='images', lbl_folder='labels'):
        super().__init__()
        self.imgs = sorted(glob.glob(os.path.join(root, img_folder, '*.png')))
        self.lbls = [p.replace(img_folder, lbl_folder) for p in self.imgs]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        # 1. 读灰度图
        img = cv2.imread(self.imgs[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
        H,W = img.shape
        img = img[None]  # [1,H,W]

        # 2. 读多分类 mask（0/1/2）
        m = cv2.imread(self.lbls[i], cv2.IMREAD_GRAYSCALE).astype(np.int64)
        # 3. 切成两路二分类 mask
        #    head1: 大血管 vs 背景  => mask_large ∈ {0,1}
        #    head2: 小血管 vs 背景  => mask_small ∈ {0,1}
        mask_large = (m==1).astype(np.float32)[None]  # [1,H,W]
        mask_small = (m==2).astype(np.float32)[None]  # [1,H,W]

        return (
            torch.from_numpy(img), 
            torch.from_numpy(mask_large),
            torch.from_numpy(mask_small)
        )

#%% 导入双通道数据集
class TwoChannelTwoHeadDataset(Dataset):
    """
    读取由 np.savez_compressed 保存的双通道 patch:
      - cd31: uint8, [H,W]
      - dapi: uint8, [H,W]
      - lbl : uint8, [H,W]，值 ∈ {0,1,2}

    返回：
      img:       FloatTensor [2, H, W] (cd31, dapi) 归一化到 [0,1]
      mask_large:FloatTensor [1, H, W] 二分类 (1=大血管)
      mask_small:FloatTensor [1, H, W] 二分类 (1=小血管)
    """
    def __init__(self, patch_folder):
        super().__init__()
        self.files = sorted(glob.glob(f"{patch_folder}/*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        cd31 = data['cd31'].astype(np.float32) / 255.0
        dapi = data['dapi'].astype(np.float32) / 255.0
        lbl  = data['lbl'].astype(np.int64)    # 0=背景, 1=大血管, 2=小血管

        # 双通道拼接
        img = np.stack([cd31, dapi], axis=0)   # [2,H,W]

        # 拆成两个二分类 mask
        mask_large = (lbl == 1).astype(np.float32)[None, ...]  # [1,H,W]
        mask_small = (lbl == 2).astype(np.float32)[None, ...]  # [1,H,W]

        return (
            torch.from_numpy(img),        # float32, [2,H,W]
            torch.from_numpy(mask_large), # float32, [1,H,W]
            torch.from_numpy(mask_small)  # float32, [1,H,W]
        )

#%% training
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
train_dir = "/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel_CGL/data/train"
save_dir = "/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/twoChannel_CGL"

os.makedirs(save_dir, exist_ok = True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 创建 GradScaler，用于动态缩放 loss
scaler     = GradScaler()
best_acc   = 0.0
patience   = 7
no_imp     = 0
epochs     = 30
lr         = 1e-4
batch_size = 32


# 训练集和验证集的划分
ds = TwoChannelTwoHeadDataset(train_dir)
n_val = int(len(ds)*0.1)
n_train = len(ds)-n_val
train_ds,val_ds = random_split(ds,[n_train, n_val], generator = torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True,  num_workers = 0, pin_memory = False)
val_loader   = DataLoader(val_ds,   batch_size = batch_size, shuffle = False, num_workers = 0, pin_memory = False)

net = MultiHeadUNet(in_ch = 2).to(device)
crit = nn.BCEWithLogitsLoss()
opt  = optim.Adam(net.parameters(), lr = lr)

# 用于记录
train_loss_large = []
train_loss_small = []
val_acc_large   = []
val_acc_small   = []

for epoch in range(1, epochs + 1):
    net.train()
    L_large = 0.0
    L_small = 0.0

    for img, m1, m2 in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Train"):
        img, m1, m2 = img.to(device), m1.to(device), m2.to(device)
        opt.zero_grad()

        # 混合精度前向 + 计算 loss
        with autocast():
            out1, out2 = net(img)
            loss1 = crit(out1, m1)
            loss2 = crit(out2, m2)
            loss  = loss1 + loss2

        # 缩放后反向
        scaler.scale(loss).backward()
        # scaler 代替 optimizer.step()
        scaler.step(opt)
        scaler.update()

        L_large += loss1.item()
        L_small += loss2.item()

    train_loss_large.append(L_large / len(train_loader))
    train_loss_small.append(L_small / len(train_loader))

    # —— 验证阶段 —— #
    net.eval()
    corr1 = tot1 = corr2 = tot2 = 0

    with torch.no_grad():
        for img, m1, m2 in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} Val"):
            img, m1, m2 = img.to(device), m1.to(device), m2.to(device)
            # 混合精度前向
            with autocast():
                o1, o2 = net(img)
            p1 = (torch.sigmoid(o1) > 0.5).float()
            p2 = (torch.sigmoid(o2) > 0.5).float()

            corr1 += (p1 == m1).sum().item()
            tot1   += m1.numel()
            corr2 += (p2 == m2).sum().item()
            tot2   += m2.numel()

    acc1 = corr1 / tot1
    acc2 = corr2 / tot2
    val_acc_large.append(acc1)
    val_acc_small.append(acc2)

    print(f"Epoch {epoch} → train_loss_large: {train_loss_large[-1]:.4f}, "
          f"train_loss_small: {train_loss_small[-1]:.4f} | "
          f"val_acc_large: {acc1:.4f}, val_acc_small: {acc2:.4f}")

    # Early stopping based on sum of acc
    if (acc1+acc2) > best_acc:
        best_acc = acc1+acc2
        no_imp   = 0
        torch.save(net.state_dict(), os.path.join(save_dir, "best_multihead.pth"))
    else:
        no_imp += 1
        if no_imp >= patience:
            print("Early stopping")
            break


#%% 绘图
epochs = list(range(1, len(train_loss_large) + 1))
plt.figure(figsize = (12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_large, 'o-', label = "Loss Large")
plt.plot(epochs, train_loss_small, 'o-', label = "Loss Small")
plt.title("Training Loss per Head")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(epochs, val_acc_large, 'o-', label = "Acc Large")
plt.plot(epochs, val_acc_small, 'o-', label = "Acc Small")
plt.title("Validation Accuracy per Head")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0,1.0)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "multihead_training_curves.png"))
plt.close()
