import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_average_channels(image, area):
    x1, y1 = int(area[0][0]), int(area[0][1])
    x2, y2 = int(area[2][0]), int(area[2][1])

    roi = image[y1:y2, x1:x2]

    average_rgb = np.mean(roi, axis=(0, 1))
    average_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).mean(axis=(0, 1))
    average_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).mean(axis=(0, 1))

    return average_rgb, average_hsv, average_lab

def process_images(folder_path, output_csv):
    file_list = os.listdir(folder_path)
    num_images = len(file_list)

    data = {
        'image_name': [],
        'organ': [],
        'RGB_R': [],
        'RGB_G': [],
        'RGB_B': [],
        'HSV_H': [],
        'HSV_S': [],
        'HSV_V': [],
        'LAB_L': [],
        'LAB_A': [],
        'LAB_B': []
    }

    for file_name in tqdm(file_list, desc='Processing Images', unit='image'):
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)

        # Define different areas for different organs
        heart_area = [(0, 0), (image.shape[1], 0), (image.shape[1], int(image.shape[0] * 0.2)), (0, int(image.shape[0] * 0.2))]
        kidney_area = [(0, int(image.shape[0] * 0.8)), (image.shape[1], int(image.shape[0] * 0.8)), (image.shape[1], image.shape[0]), (0, image.shape[0])]
        spleen_area = [(int(image.shape[1] * 0.2), int(image.shape[0] * 0.2)), (int(image.shape[1] * 0.8), int(image.shape[0] * 0.2)), (int(image.shape[1] * 0.8), int(image.shape[0] * 0.8)), (int(image.shape[1] * 0.2), int(image.shape[0] * 0.8))]
        lung_area = [(0, int(image.shape[0] * 0.2)), (int(image.shape[1] * 0.2), int(image.shape[0] * 0.2)), (int(image.shape[1] * 0.2), int(image.shape[0] * 0.8)), (0, int(image.shape[0] * 0.8))]
        liver_area = [(int(image.shape[1] * 0.8), int(image.shape[0] * 0.2)), (image.shape[1], int(image.shape[0] * 0.2)), (image.shape[1], int(image.shape[0] * 0.8)), (int(image.shape[1] * 0.8), int(image.shape[0] * 0.8))]

        # Compute average values for each channel (RGB, HSV, LAB) for each organ
        heart_avg_rgb, heart_avg_hsv, heart_avg_lab = compute_average_channels(image, heart_area)
        kidney_avg_rgb, kidney_avg_hsv, kidney_avg_lab = compute_average_channels(image, kidney_area)
        spleen_avg_rgb, spleen_avg_hsv, spleen_avg_lab = compute_average_channels(image, spleen_area)
        lung_avg_rgb, lung_avg_hsv, lung_avg_lab = compute_average_channels(image, lung_area)
        liver_avg_rgb, liver_avg_hsv, liver_avg_lab = compute_average_channels(image, liver_area)

        # Add data to the dictionary
        data['image_name'].extend([file_name] * 5)
        data['organ'].extend(['heart', 'kidney', 'spleen', 'lung', 'liver'])
        data['RGB_R'].extend([heart_avg_rgb[0], kidney_avg_rgb[0], spleen_avg_rgb[0], lung_avg_rgb[0], liver_avg_rgb[0]])
        data['RGB_G'].extend([heart_avg_rgb[1], kidney_avg_rgb[1], spleen_avg_rgb[1], lung_avg_rgb[1], liver_avg_rgb[1]])
        data['RGB_B'].extend([heart_avg_rgb[2], kidney_avg_rgb[2], spleen_avg_rgb[2], lung_avg_rgb[2], liver_avg_rgb[2]])
        data['HSV_H'].extend([heart_avg_hsv[0], kidney_avg_hsv[0], spleen_avg_hsv[0], lung_avg_hsv[0], liver_avg_hsv[0]])
        data['HSV_S'].extend([heart_avg_hsv[1], kidney_avg_hsv[1], spleen_avg_hsv[1], lung_avg_hsv[1], liver_avg_hsv[1]])
        data['HSV_V'].extend([heart_avg_hsv[2], kidney_avg_hsv[2], spleen_avg_hsv[2], lung_avg_hsv[2], liver_avg_hsv[2]])
        data['LAB_L'].extend([heart_avg_lab[0], kidney_avg_lab[0], spleen_avg_lab[0], lung_avg_lab[0], liver_avg_lab[0]])
        data['LAB_A'].extend([heart_avg_lab[1], kidney_avg_lab[1], spleen_avg_lab[1], lung_avg_lab[1], liver_avg_lab[1]])
        data['LAB_B'].extend([heart_avg_lab[2], kidney_avg_lab[2], spleen_avg_lab[2], lung_avg_lab[2], liver_avg_lab[2]])

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

    print("数据已成功保存。")

folder_path = r'E:\dachuang_7_7\data\sx\tongue'  # 输入存储图像的文件夹路径
output_csv = r'E:\dachuang_7_7\data\sx\tongue_output.csv'  # 输出 CSV 文件的路径

process_images(folder_path, output_csv)