import cv2
import numpy as np
import matplotlib.pyplot as plt

def fit_and_save_curves(input_image_path, output_image_path):
    # 读取图像
    img = cv2.imread(input_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 图像预处理
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 对每个轮廓进行多项式拟合
    for contour in contours:
        if len(contour) > 50:  # 过滤掉太小的轮廓
            x = contour[:, 0, 0]
            y = contour[:, 0, 1]
            
            # 使用3次多项式拟合
            z = np.polyfit(x, y, 3)
            curve = np.poly1d(z)
            
            # 绘制拟合曲线
            x_new = np.linspace(min(x), max(x), 100)
            y_new = curve(x_new)
            
            # 将点转换为整数坐标
            pts = np.array([[int(x), int(y)] for x, y in zip(x_new, y_new)])
            pts = pts.reshape((-1, 1, 2))
            
            # 在原图上绘制曲线
            cv2.polylines(img, [pts], False, (0, 255, 0), 2)
    
    # 保存结果
    cv2.imwrite(output_image_path, img)
    print(f"结果已保存到 {output_image_path}")

# 使用函数
input_image_path = 'point3.jpg'
output_image_path = 'D:/Anaconda/envs/Train/TRAIN_ROAD/4551903cb376cac6f6a386dba595dac.png'
fit_and_save_curves(input_image_path, output_image_path)