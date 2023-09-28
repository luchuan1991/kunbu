import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def compute_average_channels(image, area):
    roi = image[area[0][1]:area[2][1], area[0][0]:area[1][0]]
    if np.any(roi):
        average_rgb = np.nanmean(roi, axis=(0, 1))
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        average_hsv = np.nanmean(roi_hsv, axis=(0, 1))
        roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        average_lab = np.nanmean(roi_lab, axis=(0, 1))
    else:
        average_rgb = np.array([0, 0, 0])
        average_hsv = np.array([0, 0, 0])
        average_lab = np.array([0, 0, 0])
    return average_rgb, average_hsv, average_lab

def process_images(image_folder):
    # 获取文件夹内所有图像文件路径
    image_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if
                   file.lower().endswith(".jpg") or file.lower().endswith(".png")]

    # 创建保存结果的DataFrame
    dfs = []

    # 使用tqdm创建进度条
    progress_bar = tqdm(total=len(image_paths), desc='Processing Images')

    for image_path in image_paths:
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            continue

        # 定义中医五脏面部分区的坐标
        heart_area = [(0, 0), (image.shape[1], 0), (image.shape[1], image.shape[0]//6), (0, image.shape[0]//6)]
        kidney_area = [(0, image.shape[0]//2), (image.shape[1], image.shape[0]//2), (image.shape[1], image.shape[0]), (0, image.shape[0])]
        spleen_area = [(image.shape[1]//3, image.shape[0]//6), (image.shape[1]//3*2, image.shape[0]//6), (image.shape[1]//3*2, image.shape[0]//2), (image.shape[1]//3, image.shape[0]//2)]
        lung_area = [(0, image.shape[0]//6),(image.shape[1]//3, image.shape[0]//6), (image.shape[1]//3, image.shape[0]//2), (0, image.shape[0]//2)]
        liver_area = [(image.shape[1]//3*2, image.shape[0]//6), (image.shape[1], image.shape[0]//6), (image.shape[1], image.shape[0]//2), (image.shape[1]//3*2, image.shape[0]//2)]

        # 计算RGB、HSV和Lab通道的平均值
        heart_average_rgb, heart_average_hsv, heart_average_lab = compute_average_channels(image, heart_area)
        kidney_average_rgb, kidney_average_hsv, kidney_average_lab = compute_average_channels(image, kidney_area)
        spleen_average_rgb, spleen_average_hsv, spleen_average_lab = compute_average_channels(image, spleen_area)
        lung_average_rgb, lung_average_hsv, lung_average_lab = compute_average_channels(image, lung_area)
        liver_average_rgb, liver_average_hsv, liver_average_lab = compute_average_channels(image, liver_area)

        # 创建临时DataFrame
        df = pd.DataFrame({
            'Image': image_path,
            'Organ': ['Heart', 'Kidney', 'Spleen', 'Lung', 'Liver'],
            'RGB_R': [int(heart_average_rgb[2]), int(kidney_average_rgb[2]), int(spleen_average_rgb[2]), int(lung_average_rgb[2]), int(liver_average_rgb[2])],
            'RGB_G': [int(heart_average_rgb[1]), int(kidney_average_rgb[1]), int(spleen_average_rgb[1]), int(lung_average_rgb[1]), int(liver_average_rgb[1])],
            'RGB_B': [int(heart_average_rgb[0]), int(kidney_average_rgb[0]), int(spleen_average_rgb[0]), int(lung_average_rgb[0]), int(liver_average_rgb[0])],
            'HSV_H': [int(heart_average_hsv[0]), int(kidney_average_hsv[0]), int(spleen_average_hsv[0]), int(lung_average_hsv[0]), int(liver_average_hsv[0])],
            'HSV_S': [int(heart_average_hsv[1]), int(kidney_average_hsv[1]), int(spleen_average_hsv[1]), int(lung_average_hsv[1]), int(liver_average_hsv[1])],
            'HSV_V': [int(heart_average_hsv[2]), int(kidney_average_hsv[2]), int(spleen_average_hsv[2]), int(lung_average_hsv[2]), int(liver_average_hsv[2])],
            'Lab_L': [int(heart_average_lab[0]), int(kidney_average_lab[0]), int(spleen_average_lab[0]), int(lung_average_lab[0]), int(liver_average_lab[0])],
            'Lab_a': [int(heart_average_lab[1]), int(kidney_average_lab[1]), int(spleen_average_lab[1]), int(lung_average_lab[1]), int(liver_average_lab[1])],
            'Lab_b': [int(heart_average_lab[2]), int(kidney_average_lab[2]), int(spleen_average_lab[2]), int(lung_average_lab[2]), int(liver_average_lab[2])]
        })

        dfs.append(df)

        # 更新进度条
        progress_bar.update(1)

    # 合并所有DataFrame
    result_df = pd.concat(dfs, ignore_index=True)

    # 保存结果为CSV文件
    csv_path = os.path.join(image_folder, 'output.csv')
    result_df.to_csv(csv_path, index=False)

    # 完成后关闭进度条
    progress_bar.close()

def main():
    image_folder = r'E:\dachuang_7_7\data\sx\face'
    process_images(image_folder)

if __name__ == '__main__':
    main()