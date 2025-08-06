# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.style.use('seaborn-v0_8-muted')
plt.rcParams['font.serif'] = ['Times New Roman']

font      = FontProperties(size=6,  family='Times New Roman')
tick_font = FontProperties(size=5,  family='Times New Roman')

# ---------- 1. 维度 ----------
labels = ['rFID', 'PSNR', 'SSIM', 'MME-P', 'MMVet',
          'VQAv2', 'GQA', 'MMMU', 'TextVQA', 'MMBench']
N = len(labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False)

# ---------- 2. 原始数据 ----------
raw_data = {
    # 'LLaVA v1.5'          : [0.15, 25.0, 0.85, 1525, 45.0, 83.0, 75.0, 50.0, 61.0, 57.0],
    'VILA-U'           : [0.80, 23.0, 0.82, 1450, 40.0, 70.0, 58.0, 45.0, 40.0, 35.0],
    'QLIP'           : [1.20, 24.5, 0.83, 1375, 41.5, 60.0, 61.0, 47.0, 37.5, 37.0],
    'UniTok' : [0.10, 26.0, 0.87, 1465, 48.0, 85.0, 77.0, 52.0, 63.0, 59.0],
    'TokenFlow'  : [0.70, 24.0, 0.84, 1500, 42.0, 75.0, 70.0, 49.0, 55.0, 50.0],
    'UniFlow'  : [0.05, 27.0, 0.88, 1550, 50.0, 88.0, 80.0, 55.0, 65.0, 62.0],
}

raw_mtx     = np.array(list(raw_data.values()))
max_per_dim = raw_mtx.max(axis=0)
min_per_dim = raw_mtx.min(axis=0)

# ---------- 3. 归一化 ----------
def normalize(vals):
    vals = np.asarray(vals, dtype=float)
    norm = np.empty_like(vals)
    # rFID 反向
    norm[0] = (max_per_dim[0] - vals[0]) / (max_per_dim[0] - min_per_dim[0]) * 100
    # 其余正向
    for k in range(1, len(vals)):
        norm[k] = (vals[k] - min_per_dim[k]) / (max_per_dim[k] - min_per_dim[k]) * 100
    return np.clip(norm, 0, 100)

# ---------- 4. 画图 ----------
fig = plt.figure(figsize=(4, 4))
fig.set_facecolor('#FFFFFF')
ax = fig.add_subplot(111, polar=True)

R_MAX = 300                       # <── 放大后的半径
ax.set_rmax(R_MAX)                # <── 告诉坐标轴
ax.set_rticks([])                 # 隐藏默认 0-100 数字

# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 画 3 条同心圆（40 %、80 %、100 %）
circle_r = np.array([0.4, 0.8, 1.0]) * R_MAX   # <── 按新比例
for r in circle_r:
    ax.plot(np.linspace(0, 2*np.pi, 200), [r]*200,
            color='grey', lw=0.1, alpha=0.4)

ax.xaxis.grid(True, color='grey', linewidth=0.1, alpha=0.4)   # ← 控制射线粗细
# ---------- 5. 画折线 ----------
lines = []
for (name, raw), color in zip(raw_data.items(), colors):
    norm = normalize(raw) * R_MAX / 100          # <── 缩放到 0-R_MAX
    norm = np.concatenate([norm, [norm[0]]])
    line, = ax.plot(np.concatenate([angles, [angles[0]]]), norm,
                    linewidth=0.75, color=color, label=name)
    lines.append(line)
    ax.fill(np.concatenate([angles, [angles[0]]]), norm,
            alpha=0.1, color=color)

# ---------- 6. 写刻度（仅 40 %、80 % 两层） ----------
pct2write = [0.4, 0.8]
for k, angle in enumerate(angles):
    lo, hi = min_per_dim[k], max_per_dim[k]
    for p in pct2write:
        val = hi - p*(hi-lo) if k==0 else lo + p*(hi-lo)
        txt = f'{val:.2f}'.rstrip('0').rstrip('.')
        ax.text(angle, p*R_MAX, txt, color='grey',
                fontproperties=tick_font, ha='center', va='center')

# ---------- 7. 装饰 ----------
ax.set_thetagrids(angles*180/np.pi, labels, fontproperties=font)
ax.spines['polar'].set_visible(False)
ax.set_yticklabels([])              # 隐藏 0-100 数字
# ax.set_title('xxxxx', fontsize=8, y=-0.25, fontproperties=font)

ax.legend(handles=lines,
          loc='lower right',      # 关键：锚点对齐方式
          ncol=1,                 # 一列更紧凑，可保持 ncol=2 也可以
          bbox_to_anchor=(1.1, -0.1),  # 右下外一点，数值可微调
          fancybox=True,
          shadow=False,
          frameon=True,
          prop=font,
          framealpha=0.1)

plt.tight_layout()
plt.savefig('radar_fig.pdf', dpi=500, bbox_inches='tight')
plt.show()
