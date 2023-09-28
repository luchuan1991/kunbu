import cv2
import csv
import numpy as np

# 读取图片
image_path = r'E:\dachuang_7_7\data\sx\face\122.jpg'
image = cv2.imread(image_path)

# 定义各个区域的坐标点
heart_area = [(0, 0), (image.shape[1], 0), (image.shape[1], image.shape[0]//6), (0, image.shape[0]//6)]
kidney_area = [(0, image.shape[0]//2), (image.shape[1], image.shape[0]//2), (image.shape[1], image.shape[0]), (0, image.shape[0])]
spleen_area = [(image.shape[1]//3, image.shape[0]//6), (image.shape[1]//3*2, image.shape[0]//6), (image.shape[1]//3*2, image.shape[0]//2), (image.shape[1]//3, image.shape[0]//2)]
lung_area = [(0, image.shape[0]//6),(image.shape[1]//3, image.shape[0]//6), (image.shape[1]//3, image.shape[0]//2), (0, image.shape[0]//2)]
liver_area = [(image.shape[1]//3*2, image.shape[0]//6), (image.shape[1], image.shape[0]//6), (image.shape[1], image.shape[0]//2), (image.shape[1]//3*2, image.shape[0]//2)]

# 绘制红线区分五脏
cv2.polylines(image, [np.array(heart_area)], True, (0, 0, 255), 8)
cv2.polylines(image, [np.array(kidney_area)], True, (0, 0, 255), 8)
cv2.polylines(image, [np.array(spleen_area)], True, (0, 0, 255), 8)
cv2.polylines(image, [np.array(lung_area)], True, (0, 0, 255), 8)
cv2.polylines(image, [np.array(liver_area)], True, (0, 0, 255), 8)

# # 在相应区域写上英文
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(image, 'Heart', (int(image.shape[1]*0.5), int(image.shape[0]*0.1)), font, 3, (0, 0, 255), 2, cv2.LINE_AA)
# cv2.putText(image, 'Kidney', (int(image.shape[1]*0.1), int(image.shape[0]*0.65)), font, 3, (0, 0, 255), 2, cv2.LINE_AA)
# cv2.putText(image, 'Spleen', (int(image.shape[1]*0.5), int(image.shape[0]*0.35)), font, 3, (0, 0, 255), 2, cv2.LINE_AA)
# cv2.putText(image, 'Lung', (int(image.shape[1]*0.2), int(image.shape[0]*0.46)), font, 3, (0, 0, 255), 2, cv2.LINE_AA)
# cv2.putText(image, 'Liver', (int(image.shape[1]*0.8), int(image.shape[0]*0.46)), font, 3, (0, 0, 255), 2, cv2.LINE_AA)

# 添加文字标注
cv2.putText(image, 'Heart', tuple(np.mean(heart_area, axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
cv2.putText(image, 'Liver', tuple(np.mean(liver_area, axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
cv2.putText(image, 'Spleen', tuple(np.mean(spleen_area, axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255),
            2)
cv2.putText(image, 'Lung', tuple(np.mean(lung_area, axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
cv2.putText(image, 'Kidney', tuple(np.mean(kidney_area, axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255),
            2)

# 获取屏幕分辨率
screen_res = 1080, 720

# 计算图片缩放比例
scale_width = screen_res[0] / image.shape[1]
scale_height = screen_res[1] / image.shape[0]
scale = min(scale_width, scale_height)

# 计算缩放后的新尺寸
window_width = int(image.shape[1] * scale)
window_height = int(image.shape[0] * scale)

# 调整窗口大小
cv2.namedWindow('Organ Segmentation', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Organ Segmentation', window_width, window_height)

# 显示图像
cv2.imshow('Organ Segmentation', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # 将图片保存为 JPG 格式
# output_path = r'E:\dachuang_7_7\data\sx\face\122output.jpg'
# cv2.imwrite(output_path, image)