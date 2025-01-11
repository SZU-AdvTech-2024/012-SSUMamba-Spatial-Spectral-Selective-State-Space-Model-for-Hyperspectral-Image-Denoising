import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def main():
    mat_file ='tree_0822-0853.mat'
    # 加载噪声数据
    noise_path = '/data/liuliyuan/Code/SSUMamba-main/justForTest/95/'+mat_file
    mat_contents = loadmat(noise_path)
    hyperspectral_data = mat_contents['DataCube']
    hyperspectral_data = np.transpose(hyperspectral_data, (2, 0, 1))
    print(f"Data shape (noise): {hyperspectral_data.shape}")
    band1 = hyperspectral_data[30, :, :]
    band2 = hyperspectral_data[20, :, :]
    band3 = hyperspectral_data[10, :, :]
    pseudo_rgb_noise = np.stack([band1, band2, band3], axis=-1)
    
    # 加载去噪数据
    denoise_path = './data/tests/mamba/44_48.55_SSCS_icvl_icvl_95_UniformNoise-mi0-ma95_0.0003/mat/'+mat_file
    mat_contents = loadmat(denoise_path)
    hyperspectral_data = mat_contents['ssumamba']
    print(f"Data shape (denoise): {hyperspectral_data.shape}")
    band1 = hyperspectral_data[30, :, :]
    band2 = hyperspectral_data[20, :, :]
    band3 = hyperspectral_data[10, :, :]
    pseudo_rgb_denoise = np.stack([band1, band2, band3], axis=-1)
    
    # 加载 GT 数据
    gt_path = '/data/liuliyuan/Code/SSUMamba-main/justForTest/ICVL/gt_minmax/'+mat_file
    mat_contents = loadmat(gt_path)
    hyperspectral_data = mat_contents['DataCube']
    hyperspectral_data = np.transpose(hyperspectral_data, (2, 0, 1))
    print(f"Data shape (GT): {hyperspectral_data.shape}")
    band1 = hyperspectral_data[30, :, :]
    band2 = hyperspectral_data[20, :, :]
    band3 = hyperspectral_data[10, :, :]
    pseudo_rgb_gt = np.stack([band1, band2, band3], axis=-1)
    
    # 创建一个包含 1 行 3 列的子图布局
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # 绘制噪声图像
    axs[0].imshow(pseudo_rgb_noise)
    axs[0].set_title('Noise')
    axs[0].axis('off')
    
    # 绘制去噪图像
    axs[1].imshow(pseudo_rgb_denoise)
    axs[1].set_title('Denoise')
    axs[1].axis('off')
    
    # 绘制 GT 图像
    axs[2].imshow(pseudo_rgb_gt)
    axs[2].set_title('GT')
    axs[2].axis('off')
    
    # 保存整合后的图像
    plt.savefig('./figs/combined_image.png')
    plt.close()


if __name__ == "__main__":
    main()