import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patheffects import Stroke, Normal
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# 设置随机种子以保证结果的可重复性
RS = 20150101
np.random.seed(RS)
torch.manual_seed(RS)

# 加载 MNIST 数据集
transform = ToTensor()
mnist_dataset = MNIST(root='.mnist_data', train=True, download=True, transform=transform)

# 提取数据和标签
X = []
y = []
for image, label in mnist_dataset:
    X.append(image.view(-1).numpy())  # 将图像展平为一维数组
    y.append(label)

X = np.array(X)
y = np.array(y)

# 为了加快 t-SNE 的运行速度，我们只使用部分数据
# 如果需要使用全部数据，可以取消以下两行的注释
X = X[:5000]
y = y[:5000]

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=RS)
digits_proj = tsne.fit_transform(X)

# 设置绘图风格
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# 绘制散点图
def scatter(x, colors):
    # 选择颜色
    palette = np.array(sns.color_palette("hls", 10))  # MNIST 有 10 个类别
    
    # 创建散点图
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int32)])
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    ax.axis('off')
    ax.axis('tight')
    
    # 添加每个类别的标签
    txts = []
    for i in range(10):  # MNIST 有 10 个类别
        # 每个标签的位置
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            Stroke(linewidth=5, foreground="w"),
            Normal()
        ])
        txts.append(txt)
    
    return f, ax, sc, txts

# 绘制 t-SNE 结果
scatter(digits_proj, y)
plt.savefig('mnist_tsne-generated.png', dpi=120)
plt.show()
