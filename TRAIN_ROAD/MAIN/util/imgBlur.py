import cv2
import numpy as np

'''gray滤波效果快速选择'''
frame = cv2.imread('/home/q/public security/photo/11.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#无滤波
cv2.imshow("gray", gray)
cv2.waitKey(0)
gray_th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 4)
cv2.imshow("gray_th", gray_th)
cv2.waitKey(0)


#高斯
Gaus_imgBlur = cv2.GaussianBlur(gray, (5, 5), 1)
cv2.imshow("GaussianBlur", Gaus_imgBlur)
cv2.waitKey(0)
gaus_th = cv2.adaptiveThreshold(Gaus_imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 4)
cv2.imshow("gaus_th", gaus_th)
cv2.waitKey(0)


#中值
Median_imgBlur = cv2.medianBlur(gray, 5)
cv2.imshow("MedianBlur", Median_imgBlur)
cv2.waitKey(0)
Median_th = cv2.adaptiveThreshold(Median_imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 4)
cv2.imshow("Median_th", Median_th)
cv2.waitKey(0)


#均值
Mean_imgBlur = cv2.blur(gray, (5, 5))
cv2.imshow("MeanBlur", Mean_imgBlur)
cv2.waitKey(0)
Mean_th = cv2.adaptiveThreshold(Mean_imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 4)
cv2.imshow("Mean_th", Mean_th)
cv2.waitKey(0)


#双边(保留颜色信息)
Bilateral_imgBlur = cv2.bilateralFilter(gray, 9, 75, 75)
cv2.imshow("Bilateral", Bilateral_imgBlur)
cv2.waitKey(0)
Bilateral_th = cv2.adaptiveThreshold(Bilateral_imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 4)
cv2.imshow("Bilateral_th", Bilateral_th)
cv2.waitKey(0)
