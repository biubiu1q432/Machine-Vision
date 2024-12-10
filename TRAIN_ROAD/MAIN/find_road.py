import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os


# '''分割出一片--》opencv划分'''


model_path = r"D:\Anaconda\envs\YOLO\TRAIN_ROAD\runs\segment\train2\weights\best.pt"
source_video_path =r"D:\Anaconda\envs\YOLO\TRAIN_ROAD\data\part-2.mp4"
source_img_path =r"D:\Anaconda\envs\YOLO\TRAIN_ROAD\data\20.jpg"

model = YOLO(model_path)  # 预训练的 YOLOv8n 模型
# 对一系列图像执行批量推理
results = model.predict(source_img_path,show=True)  # 返回一个结果对象列表

#数据处理
frame = cv2.imread(source_img_path)

for result in results:
	if result.masks is not None:
		mask_raw = result.masks[0].cpu().data.numpy().transpose(1,2,0)
        for mask in result.masks:
            mask_raw += mask.cpu().data.numpy().transpose(1,2,0)

cv2.imshow("mask", mask_raw)
cv2.waitKey(0)




# model_path = r"D:\Anaconda\envs\YOLO\TRAIN_ROAD\runs\segment\train2\weights\best.pt"
# source_video_path =r"D:\Anaconda\envs\YOLO\TRAIN_ROAD\data\part-2.mp4"
# source_img_path =r"D:\Anaconda\envs\YOLO\TRAIN_ROAD\data\20.jpg"

# model = YOLO(model_path)  # 预训练的 YOLOv8n 模型
# results = model.predict(source_img_path,show=True)  # 返回一个结果对象列表

# frame = cv2.imread(source_img_path)

# #创建与frame同样大小的黑色图片
# mask = np.zeros_like(frame)

# #获取掩码边缘坐标    
# for result in results:
#     masks = result.masks       

# masks_xy = masks.xy    # 每个掩码的边缘点坐标
# masks_xy = [np.array(mask).astype(int) for mask in masks_xy]

# print(type(masks_xy))

# #把在掩码内的像素点设为白色
# for mask_xy in masks_xy:
#     cv2.fillPoly(mask, [mask_xy], (255, 255, 255))

# cv2.imshow("mask", mask)
# cv2.waitKey(0)








# def is_point_inside_polygon(x, y, polygon):
#     n = len(polygon)
#     inside = False
#     j = n - 1
#     for i in range(n):
#         if ((polygon[i][1] > y) != (polygon[j][1] > y)) and \
#            (x < polygon[i][0] + (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1])):
#             inside = not inside
#         j = i
#     return inside
 
# def find_polygon_pixels(masks_xy, boxes_cls):    # 所有掩码像素点及其对应类别属性的列表
#     # 初始化存储所有像素点和类别属性的列表
#     all_pixels_with_cls = []
 
#     # 遍历每个多边形
#     for i, polygon in enumerate(masks_xy):
#         cls = boxes_cls[i]  # 当前多边形的类别属性
 
#         # 将浮点数坐标点转换为整数类型
#         polygon = [(int(point[0]), int(point[1])) for point in polygon]
 
#         # 找出当前多边形的边界框
#         min_x = min(point[0] for point in polygon)
#         max_x = max(point[0] for point in polygon)
#         min_y = min(point[1] for point in polygon)
#         max_y = max(point[1] for point in polygon)
 
#         # 在边界框内遍历所有像素点
#         for x in range(min_x, max_x + 1):
#             for y in range(min_y, max_y + 1):
#                 # 检查像素点是否在多边形内部
#                 if is_point_inside_polygon(x, y, polygon):
#                     # 将像素点坐标和类别属性组合成元组，添加到列表中
#                     all_pixels_with_cls.append(((x, y), cls))
 
#     return all_pixels_with_cls
 
# def reconstruct_image(image_size, pixels_with_cls):
#     # 创建一个和图片原始大小相同的黑色图像
#     reconstructed_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
 
#     # 将属性为 0 的像素点设为绿色，属性为 1 的像素点设为蓝色 ，其余的像素点默认为背景设为黑色
#     for pixel, cls in pixels_with_cls:
#         if cls == 0:
#             reconstructed_image[pixel[1], pixel[0]] = [0, 255, 0]  # 绿色
#         elif cls == 1:
#             reconstructed_image[pixel[1], pixel[0]] = [0, 0, 255]  # 蓝色
#         else:
#             reconstructed_image[pixel[1], pixel[0]] = [0, 0, 0]    # 黑色
 
#     return reconstructed_image


# # 获取RGB图像的路径
# image_path = r"D:\Anaconda\envs\YOLO\TRAIN_ROAD\data\30.jpg"

# # 执行模型预测
# model = YOLO(r"D:\Anaconda\envs\YOLO\TRAIN_ROAD\runs\segment\train2\weights\best.pt" )
# results = model(image_path,show=True)
# image = Image.open(image_path)

# # 提取掩码和检测框信息
# for result in results:
#     boxes = result.boxes          # 输出的检测框
#     masks = result.masks          # 输出的掩码信息

# masks_xy = masks.xy    # 每个掩码的边缘点坐标
# boxes_cls = boxes.cls  # 每个多边形的类别属性

# # 调用函数找出每个mask内部的点和相应的类别属性
# all_pixels_with_cls = find_polygon_pixels(masks_xy, boxes_cls)


# print(all_pixels_with_cls)


# # 对每一张图像的分割掩码进行重建并保存在特定的文件夹中
# image_size = image.size

# reconstructed_image = reconstruct_image(image_size, all_pixels_with_cls)  # 重建图像

# cv2.imwrite("reconstructed_image.jpg", reconstructed_image)

# # 显示重建的图像
# cv2.imshow("reconstructed_image", reconstructed_image)
# cv2.waitKey(0)



