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
from sklearn.model_selection import KFold
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_dir = "/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/HML_model/data/train"
save_dir = "/public3/Xinyu/3D_tissue/IF/Vascular_IF/MultiClassification_model/multiHead/HML_model/K-fold"

os.makedirs(save_dir, exist_ok = True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 创建 GradScaler，用于动态缩放 loss
scaler     = GradScaler()
best_acc   = 0.0
patience   = 7
no_imp     = 0
epochs     = 10
lr         = 1e-4
batch_size = 32
k_folds    = 10

# 加载整个数据集
dataset = TwoChannelTwoHeadDataset(train_dir)

# KFold 划分
kfold = KFold(n_splits = k_folds, shuffle = True, random_state = 42)

# 用来存每一折三类的最佳精度
fold_acc_bg    = []
fold_acc_large = []
fold_acc_small = []


for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset), 1):
    print(f"\n=== Fold {fold}/{k_folds} ===")
    # 子集 & DataLoader
    train_sub = torch.utils.data.Subset(dataset, train_idx)
    val_sub   = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_sub,   batch_size=batch_size, shuffle=False, num_workers=4)

    # 重置模型、优化器、scaler
    net    = MultiHeadUNet(in_ch=2).to(device)
    opt    = optim.Adam(net.parameters(), lr=lr)
    crit   = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    best_acc = 0.0
    no_imp   = 0

    for epoch in range(1, epochs+1):
        net.train()
        total_l1 = total_l2 = 0.0
        for img, m1, m2 in tqdm(train_loader, desc=f"Fold{fold} Epoch {epoch}"):
            img, m1, m2 = img.to(device), m1.to(device), m2.to(device)
            opt.zero_grad()
            with autocast():
                o1, o2 = net(img)
                l1 = crit(o1, m1); l2 = crit(o2, m2)
                loss = l1 + l2
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            total_l1 += l1.item(); total_l2 += l2.item()

        # 验证阶段
    net.eval()
    # 用于累积三类正确数和总数
    corr_bg = corr_l = corr_s = 0
    tot_bg  = tot_l  = tot_s  = 0

    with torch.no_grad():
        for img, m1, m2 in val_loader:
            img, m1, m2 = img.to(device), m1.to(device), m2.to(device)
            with autocast():
                o1, o2 = net(img)
            p1 = (torch.sigmoid(o1)>0.5).float()
            p2 = (torch.sigmoid(o2)>0.5).float()

            # 构造三分类真值和预测
            # m1==1 → 大血管；m2==1 → 小血管；否则背景
            true = torch.zeros_like(m1, dtype=torch.long)
            true[m1==1] = 1
            true[m2==1] = 2

            # 预测规则：先判为大血管，再判为小血管，其余背景
            pred = torch.zeros_like(true)
            pred[p1==1] = 1
            mask_small_only = (p1==0) & (p2==1)
            pred[mask_small_only] = 2

            # 展平
            t = true.view(-1)
            p = pred.view(-1)

            # 累计每一类
            for cls, (c_corr, c_tot) in enumerate([
                    (corr_bg, tot_bg),
                    (corr_l,   tot_l),
                    (corr_s,   tot_s),
                ]):
                mask = (t==cls)
                c_tot += mask.sum().item()
                c_corr += (p[mask]==cls).sum().item()

                if cls==0:
                    corr_bg, tot_bg = c_corr, c_tot
                elif cls==1:
                    corr_l, tot_l   = c_corr, c_tot
                else:
                    corr_s, tot_s   = c_corr, c_tot
        # 计算三类准确率
        acc_bg    = corr_bg / tot_bg
        acc_l     = corr_l  / tot_l
        acc_s     = corr_s  / tot_s
        print(f"Fold{fold} → 背景 acc: {acc_bg:.4f}, 大血管 acc: {acc_l:.4f}, 小血管 acc: {acc_s:.4f}")

        fold_acc_bg.append(acc_bg)
        fold_acc_large.append(acc_l)
        fold_acc_small.append(acc_s)

print("\n=== Cross Validation Done ===")
print("背景平均 acc: ", np.mean(fold_acc_bg))
print("大血管平均 acc: ", np.mean(fold_acc_large))
print("小血管平均 acc: ", np.mean(fold_acc_small))


#%% 绘图：三条折线展示三类在每折上的准确率
import matplotlib.pyplot as plt

folds = np.arange(1, k_folds+1)
plt.figure(figsize=(8,5))
plt.plot(folds, fold_acc_bg,    'o-', label='背景准确率')
plt.plot(folds, fold_acc_large, 'o-', label='大血管准确率')
plt.plot(folds, fold_acc_small, 'o-', label='小血管准确率')

# 平均值虚线
plt.axhline(np.mean(fold_acc_bg),    color='C0', linestyle='--', label=f'背景平均 {np.mean(fold_acc_bg):.3f}')
plt.axhline(np.mean(fold_acc_large), color='C1', linestyle='--', label=f'大血管平均 {np.mean(fold_acc_large):.3f}')
plt.axhline(np.mean(fold_acc_small), color='C2', linestyle='--', label=f'小血管平均 {np.mean(fold_acc_small):.3f}')

plt.xticks(folds)
plt.ylim(0,1.0)
plt.xlabel('Fold 折数')
plt.ylabel('分类准确率')
plt.title('三分类（背景/大血管/小血管）在各折上的准确率')
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'three_class_kfold_acc.png'), dpi=200)
plt.show()
