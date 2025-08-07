import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA

# ------------------------- 1. 参数 -------------------------
patch_h = 32
patch_w = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------- 2. 加载模型 ----------------------
model_name = "/fs-computility/video/shared/hf_weight/dinov2-base"   # 或自己本地路径
model = AutoModel.from_pretrained(model_name)
# 不一定所有模型都可以实现的哦

# ------------------------- 3. 读取并预处理图片 --------------
# 用 HF 自带的 processor，无需自己写 transform
image_transforms = T.Compose([
    T.Resize(448, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(448),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

images = [image_transforms(Image.open(f"/fs-computility/video/shared/donglu/Projects/yzr/OmniTok/visualization/images/{k+1}.jpg").convert("RGB"))
          for k in range(4)]

images = torch.stack(images, dim=0)

with torch.no_grad():
    outputs = model(images)
    # [B, 1 + patch_h * patch_w, feat_dim]  -> 去掉 cls
    patch_tokens = outputs.last_hidden_state[:, 1:]          # [B, N, C]

    B, N, C = patch_tokens.shape
    features = patch_tokens.reshape(B * N, C).cpu().numpy()     # [B*N, C]

# ------------------------- 4. PCA ---------------------------
pca = PCA(n_components=3)
print(features.shape)
pca_features = pca.fit_transform(features)   # [B*N, 3]
print(pca_features.shape)

# ------------------------- 5. 可视化 ------------------------
# 5.1 每个通道的直方图
plt.figure(figsize=(12, 3))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.hist(pca_features[:, i], bins=50)
    plt.title(f"PCA-{i}")
plt.tight_layout()
plt.show()

# 5.2 用第一主成分做前景/背景分割
pca_bg  = pca_features[:, 0] < 10   # 阈值可根据直方图调整
print(pca_bg.shape)
pca_fg  = ~pca_bg

# 背景 mask 可视化
bg_mask = pca_bg.reshape(B, patch_h, patch_w)
plt.figure(figsize=(8, 4))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(bg_mask[i], cmap="gray")
    plt.axis("off")
    plt.title(f"bg mask {i+1}")
plt.tight_layout()
plt.show()

# 5.3 仅在前景上做二次 PCA（可选，但通常更干净）
if pca_fg.any():
    pca2 = PCA(n_components=3)
    pca2_features = pca2.fit_transform(features[pca_fg])
    # 归一化到 0~1
    pca2_norm = (pca2_features - pca2_features.min(axis=0)) / \
                (pca2_features.max(axis=0) - pca2_features.min(axis=0))
else:
    pca2_norm = np.zeros((pca_fg.sum(), 3))

# 5.4 组装 RGB 伪彩色图
rgb_vis = np.zeros((B * patch_h * patch_w, 3))
rgb_vis[pca_bg] = 0
rgb_vis[pca_fg] = pca2_norm
rgb_vis = rgb_vis.reshape(B, patch_h, patch_w, 3)

plt.figure(figsize=(8, 4))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(rgb_vis[i])
    plt.axis("off")
    plt.title(f"PCA RGB {i+1}")
plt.tight_layout()
plt.savefig("features.png", dpi=150)
plt.show()
