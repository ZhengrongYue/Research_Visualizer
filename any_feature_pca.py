import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from tokenizer.intern_vit_teacher import internvit_patch14_448_teacher

import torch
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')          # 终端无显示
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

# ------------------------- 1. 参数 -------------------------
patch_size = 14
img_size   = 448
patch_h    = img_size // patch_size
patch_w    = img_size // patch_size
device     = "cpu"
out_dir    = "./"              # 保存目录，可改

# ------------------------- 2. 模型 -------------------------
ckpt_path = None
ckpt_path = '/fs-computility/video/shared/donglu/Projects/yzr/OmniTok/results_ckpt/patch_flow-PatchFlow_internvit_selfidstill_fulllayer_d64_gtb/checkpoints/0075000.pt'
model = internvit_patch14_448_teacher()
if ckpt_path is not None:
  checkpoint = torch.load(ckpt_path, map_location='cpu')
  checkpoint_model = checkpoint['model']
  state_dict = model.state_dict()
  msg = model.load_state_dict(checkpoint_model, strict=False)

# ------------------------- 3. 图片预处理 --------------------
transform = T.Compose([
    T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(img_size),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

folder = "/fs-computility/video/shared/donglu/Projects/yzr/OmniTok/visualization/images"
paths  = [os.path.join(folder, f"{k+1}.jpg") for k in range(4)]
imgs   = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths], dim=0).to(device)

# ------------------------- 4. 提取 patch token --------------
with torch.no_grad():
    feats = model.forward_features(imgs)[:, 1:]   # [B, N, C]
    B, N, C = feats.shape
    feats = feats.reshape(B * N, C).cpu().numpy()

"""
调参：
1. thr_pct
2. mask = comp0 < thr： > or <
3. encoder提取feature的layer
"""

def pca_visualize(feats: np.ndarray,
                  B: int,
                  patch_h: int,
                  patch_w: int,
                  out_dir: str = ".",
                  thr_pct: float =33.0,
                  bg_black: bool = True) -> None:
    """
    feats    : [B*patch_h*patch_w, C]  numpy array
    bg_black : True  -> 背景纯黑，前景二次 PCA
               False -> 整张图（含背景）一次 PCA
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1. 第一次 PCA ----------
    pca1 = PCA(n_components=3)
    pca1.fit(feats)
    full_pca = pca1.transform(feats)

    # 直方图
    plt.figure(figsize=(12, 3))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.hist(full_pca[:, i], bins=50)
        plt.title(f"PCA-{i}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_all.png"), dpi=150)
    plt.close()

    # ---------- 2. 前景/背景分割 ----------
    comp0 = full_pca[:, 0]
    thr = np.percentile(comp0, thr_pct)
    mask = comp0 < thr                     # True = 前景

    # ---------- 3. 决定最终 RGB ----------
    rgb_vis = np.zeros((B * patch_h * patch_w, 3))

    if bg_black:
        # 背景纯黑，前景用二次 PCA
        pca2 = PCA(n_components=3)
        pca2.fit(feats[mask])
        fg_pca = pca2.transform(feats[mask])
        fg_pca = (fg_pca - fg_pca.min(0)) / (fg_pca.max(0) - fg_pca.min(0) + 1e-8)
        rgb_vis[mask] = fg_pca
    else:
        # 1) 整张图先整体 PCA（含背景）
        rgb_vis[:] = full_pca
        rgb_vis = (rgb_vis - rgb_vis.min()) / (rgb_vis.max() - rgb_vis.min() + 1e-8)

    # ---------- 4. 保存 ----------
    rgb_vis = rgb_vis.reshape(B, patch_h, patch_w, 3)
    plt.figure(figsize=(8, 4))
    for i in range(B):
        plt.subplot(2, 2, i + 1)
        plt.imshow(rgb_vis[i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,
                             "internvit_pca_blackbg.png" if bg_black else "internvit_pca_full.png"),
                dpi=150)
    plt.close()
    print(f"Done! bg_black={bg_black}, 结果已保存至 {out_dir}")

# 纯黑背景
pca_visualize(feats, B, patch_h, patch_w, out_dir="./results", bg_black=True)

# 整张图 PCA（背景也上色）
pca_visualize(feats, B, patch_h, patch_w, out_dir="./results", bg_black=False)



