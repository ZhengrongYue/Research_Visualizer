import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))  # sys.path.append(".")

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import time
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import resize
from PIL import Image
from tokenizer import models_vit_unitok  # 确保这个模块路径正确

# 设置随机种子以保证可复现性
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def extract_feature(ckpt_path, inputs, layer_name, device):
    model = models_vit_unitok.__dict__['unitok_vit_base']()
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()

    msg = model.load_state_dict(checkpoint_model, strict=False)
    # print(msg)

    model = model.to(device)
    model.eval()  # 设置为评估模式

    inputs = inputs.to(device)
    print(inputs.shape)

    feature = model.extract_feature(inputs, layer_name)
    return feature

def draw_features_channels(x, savename):  # all_channel or merge
    """
    绘制特征图并保存为图像文件。
        x (tensor): 输入的特征图张量，形状为 [batch_size, num_features, H, W]。
        savename (str): 保存图像的文件名。
    """
    tic = time.time()  # 开始计时
    num_features = x.shape[1]  # 获取特征图的数量

    # 动态调整网格布局，确保网格大小与特征图数量匹配
    height = int(np.ceil(np.sqrt(num_features)))  # 调整高度
    width = int(np.ceil(num_features / height))   # 调整宽度

    fig = plt.figure(figsize=(16, 16))  # 创建图像
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)

    for i in range(num_features):
        plt.subplot(height, width, i + 1)  # 创建子图
        plt.axis('off')  # 关闭坐标轴
        img = x[0, i, :, :].numpy()  # 提取第 i 个特征图，假设 x 是 PyTorch 张量
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)  # 归一化
        plt.imshow(img, cmap='viridis')  # gray or viridis
        print(f"{i + 1}/{num_features}")  # 打印进度

    fig.savefig(savename, dpi=100)  # 保存图像
    fig.clf()
    plt.close()
    print("Time elapsed: {:.2f} seconds".format(time.time() - tic))  # 打印耗时

def draw_features_merge(x, savename, method='mean'):
    """
    将多通道特征图合并为单通道灰度图并保存为图像文件。

    参数:
        x (tensor): 输入的特征图张量，形状为 [b=1, c, h, w]。
        savename (str): 保存图像的文件名。
        method (str): 合并通道的方法，可选 'mean'（平均法）、'max'（最大值法）或 'sum'（加权求和法）。
    """
    tic = time.time()  # 开始计时
    assert x.shape[0] == 1, "Batch size must be 1"

    # 提取特征图
    feature_maps = x[0].numpy()  # 形状为 [8, h, w]

    # 根据指定方法合并通道
    if method == 'mean':
        gray_map = np.mean(feature_maps, axis=0)  # 求平均
    elif method == 'max':
        gray_map = np.max(feature_maps, axis=0)  # 取最大值
    elif method == 'sum':
        gray_map = np.sum(feature_maps, axis=0)  # 求和
    else:
        raise ValueError("Invalid method. Choose from 'mean', 'max', or 'sum'.")

    # 归一化灰度图
    gray_map = (gray_map - np.min(gray_map)) / (np.max(gray_map) - np.min(gray_map) + 1e-6)

    # 绘制并保存图像
    plt.figure(figsize=(16, 16))
    plt.imshow(gray_map, cmap='viridis')
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(savename, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Gray feature map saved to {savename}")
    print(f"Time elapsed: {time.time() - tic:.2f} seconds")

# 主函数
if __name__ == "__main__":
    # 加载图像并转换为Tensor
    image_path = "/mnt/workspace/Project/UnderGenTokenizer/UniTok/assets/dog.jpeg"  # 替换为你的图像路径
    # image_path = '/mnt/workspace/data/imagenet/data/newtrain/kite/kite.70.JPEG'
    ckpt_path = "/mnt/workspace/Project/UnderGenTokenizer/UniTok/results_ckpt/VQ-8-UniTok_CNN_VQ_Triple_16384_10epoch_OnlyKD/checkpoints/0035000.pt"
    # ckpt_path = "/mnt/workspace/Project/UnderGenTokenizer/UniTok/ckpt/vit_base_gdq_imagenet_10epoch.pt"
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    org_h, org_w = image.height, image.width  # 获取原始图像的高和宽
    image = transform(image).unsqueeze(0)  # 添加批量维度
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 提取特征
    layer_name = 'gvq'  # gvq, post_quant_conv
    feature = extract_feature(ckpt_path, image, layer_name, device=device)[0].squeeze(2).cpu()
    feature = resize(feature, size=(org_h, org_w), interpolation=transforms.InterpolationMode.BILINEAR)
    print(feature.shape)

    # [1, c=8, 16, 16]
    if layer_name == 'gvq':
        c1_feature, c2_feature, c3_feature  = torch.split(feature, split_size_or_sections=8, dim=1)

        draw_features_channels(x=c1_feature, savename="visualization/c1_feature_maps.png")
        draw_features_channels(x =c2_feature, savename="visualization/c2_feature_maps.png")
        draw_features_channels(x=c2_feature, savename="visualization/c3_feature_maps.png")

        draw_features_merge(x=c1_feature, savename="visualization/c1_feature_maps_merge.png")
        draw_features_merge(x=c2_feature, savename="visualization/c2_feature_maps_merge.png")
        draw_features_merge(x=c2_feature, savename="visualization/c3_feature_maps_merge.png")

    else:
        feature = feature.permute(1,0,2,3)
        print(feature.shape)
        draw_features_channels(x=feature, savename="visualization/feature_maps.png")
        draw_features_merge(x=feature, savename="visualization/feature_maps_merge.png")
