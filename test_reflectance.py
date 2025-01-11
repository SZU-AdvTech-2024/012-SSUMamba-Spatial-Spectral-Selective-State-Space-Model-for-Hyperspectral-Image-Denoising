import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os


def plot_reflectance_comparison(pixel1, pixel2):
    """
    该函数用于绘制两张高光谱图像中对应像素的反射率曲线对比图。

    参数:
    pixel1 (numpy.ndarray): 第一张图像中一个像素的高光谱反射率数据，形状为 (n_bands,)
    pixel2 (numpy.ndarray): 第二张图像中一个像素的高光谱反射率数据，形状为 (n_bands,)
    """
    # 假设 pixel1 和 pixel2 是一维的 numpy 数组，包含了该像素在不同波段的反射率
    # 生成波段编号，假设波段从 0 开始编号
    bands = np.arange(len(pixel1))
    # 绘制反射率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(bands, pixel1, marker='o', linestyle='-', color='b', label='SSCS&ASSM')
    plt.plot(bands, pixel2, marker='s', linestyle='--', color='r', label='SSCS')
    plt.xlabel('Band Number')
    plt.ylabel('Reflectance')
    plt.title('Reflectance Comparison of Pixels')
    plt.legend()
    plt.grid(True)
    plt.savefig('./figs/Reflectance.png')


def process_hyperspectral_images(folder_path):
    """
    该函数用于处理指定文件夹下的所有高光谱图像。
    对每个 ASSM、SSCS 对 GT 求差的绝对值的总和汇总为 pixel1 与 pixel2。

    参数:
    folder_path (str): 包含高光谱图像的文件夹路径
    """
    pixel1_sum = 0
    pixel2_sum = 0
    num_pixels = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mat'):
                mat_file = os.path.join(root, file)
                # 加载 GT 数据
                gt_path = '/data/liuliyuan/Code/SSUMamba-main/justForTest/ICVL/gt_minmax/' + file
                mat_contents = loadmat(gt_path)
                GT_data = mat_contents['DataCube']
                print(f"Data shape (denoise): {GT_data.shape}")
                # 加载 ASSM 数据
                denoise_path = './data/tests/mamba/44_48.55_SSCS_icvl_icvl_95_UniformNoise-mi0-ma95_0.0003/mat/' + file
                mat_contents = loadmat(denoise_path)
                ASSM_data = mat_contents['ssumamba']
                ASSM_data = np.transpose(ASSM_data, (1, 2, 0))
                print(f"Data shape (denoise): {ASSM_data.shape}")
                # 加载 SSCS 数据
                denoise_path = './data/tests/mamba/SSCS_icvl_95_UniformNoise-mi0-ma95_0.0003/mat/' + file
                mat_contents = loadmat(denoise_path)
                SSCS_data = mat_contents['ssumamba']
                SSCS_data = np.transpose(SSCS_data, (1, 2, 0))
                print(f"Data shape (denoise): {SSCS_data.shape}")
                # 计算差值的绝对值
                image1 = np.abs(ASSM_data - GT_data)
                image2 = np.abs(SSCS_data - GT_data)
                # 计算差值绝对值的总和
                pixel1_sum += image1
                pixel2_sum += image2
                num_pixels += 1
    # 计算平均值
     # 计算平均值
    pixel1 = pixel1_sum
    pixel2 = pixel2_sum
    row = 3
    col = 4
    pixel1 = pixel1[row, col, :]
    pixel2 = pixel2[row, col, :]
    print(f"Data shape (denoise): {pixel1.shape}")
    plot_reflectance_comparison(pixel1, pixel2)


if __name__ == "__main__":
    folder_path = './data/tests/mamba/44_48.55_SSCS_icvl_icvl_95_UniformNoise-mi0-ma95_0.0003/mat/'
    process_hyperspectral_images(folder_path)