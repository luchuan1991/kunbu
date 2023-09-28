import dlib
import cv2
import numpy as np

# 加载dlib的人脸检测器和关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# 读取图像
image_path = r'E:\dachuang_7_7\data\sx\face\122.jpg'
image = cv2.imread(image_path)


# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 进行特征点检测
detections = detector(image)
shape = predictor(image, detections[0])

# 人脸检测
faces = detector(gray)

for face in faces:
    # 关键点检测
    landmarks = predictor(gray, face)

       # 定义特征点索引
    point_22 = 22
    point_23 = 23
    point_34 = 34
    point_41 = 41
    point_47 = 47
       # 获取特征点坐标
    point_A_coord = (0, 0)
    point_B_coord = (image.shape[1], 0)
    point_C_coord = (0,image.shape[0])
    point_D_coord = (image.shape[1],image.shape[0])
    point_E_coord = (0, shape.part(point_22).y)
    point_F_coord = (shape.part(point_41).x, shape.part(point_22).y)
    point_G_coord = (shape.part(point_47).x, shape.part(point_22).y)
    point_H_coord = (image.shape[1], shape.part(point_22).y)
    point_I_coord = (0, shape.part(point_34).y)
    point_J_coord = (shape.part(point_41).x, shape.part(point_34).y)
    point_K_coord = (shape.part(point_47).x, shape.part(point_34).y)
    point_L_coord = (image.shape[1], shape.part(point_34).y)

    # 打印特征点坐标
    print("A:", point_A_coord)
    print("B:", point_B_coord)
    print("C:", point_C_coord)
    print("D:", point_D_coord)
    print("E:", point_E_coord)
    print("F:", point_F_coord)
    print("G:", point_G_coord)
    print("H:", point_H_coord)
    print("I:", point_I_coord)
    print("J:", point_J_coord)
    print("K:", point_K_coord)
    print("L:", point_L_coord)
    # 分区线段坐标
    heart_area = [point_A_coord, point_B_coord, point_H_coord, point_E_coord]
    liver_area = [point_G_coord, point_H_coord, point_L_coord, point_K_coord]
    spleen_area = [point_F_coord, point_G_coord, point_K_coord, point_J_coord]
    lung_area = [point_E_coord, point_F_coord, point_J_coord, point_I_coord]
    kidney_area =[point_I_coord, point_L_coord, point_D_coord, point_C_coord]

    # 绘制红线区分五脏
    cv2.polylines(image, [np.array(heart_area)], True, (0, 0, 255), 8)
    cv2.polylines(image, [np.array(kidney_area)], True, (0, 0, 255), 8)
    cv2.polylines(image, [np.array(spleen_area)], True, (0, 0, 255), 8)
    cv2.polylines(image, [np.array(lung_area)], True, (0, 0, 255), 8)
    cv2.polylines(image, [np.array(liver_area)], True, (0, 0, 255), 8)

    # 添加文字标注
    cv2.putText(image, 'Heart', tuple(np.mean(heart_area, axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
    cv2.putText(image, 'Liver', tuple(np.mean(liver_area, axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
    cv2.putText(image, 'Spleen', tuple(np.mean(spleen_area, axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
    cv2.putText(image, 'Lung', tuple(np.mean(lung_area, axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
    cv2.putText(image, 'Kidney', tuple(np.mean(kidney_area, axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)


# 窗口自适应大小显示图片
win_name = "Result"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.imshow(win_name, image)
cv2.waitKey(0)
cv2.destroyAllWindows()