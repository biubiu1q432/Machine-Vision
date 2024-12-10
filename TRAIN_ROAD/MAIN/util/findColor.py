'''rgb/hsv'''

import math
import threading
import serial
import time
import cv2
import numpy as np
import sys


cap = None
start,end = 0.0,0.0
def nothing(x):
    pass
WindowName = 'result'

cv2.namedWindow(WindowName, cv2.WINDOW_KEEPRATIO)  # 建立空窗口
cv2.resizeWindow(WindowName, 200, 160)  # 调整窗口大小
cv2.createTrackbar('Bl', WindowName, 0, 255, nothing)  # 创建滑动条
cv2.createTrackbar('Gl', WindowName, 0, 255, nothing)  # 创建滑动条
cv2.createTrackbar('Rl', WindowName, 0, 255, nothing)  # 创建滑动条
cv2.createTrackbar('Bh', WindowName, 255, 255, nothing)  # 创建滑动条
cv2.createTrackbar('Gh', WindowName, 255, 255, nothing)  # 创建滑动条
cv2.createTrackbar('Rh', WindowName, 255, 255, nothing)  # 创建滑动条
cv2.createTrackbar('iterations', WindowName, 0, 20, nothing)  # 创建滑动条


def main():
    global cap
    for i in  range(0,100):
        try:
            cap = cv2.VideoCapture(i)
            print("cap on\n") 
            break
        except:
            pass  
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    while True:
            global start,end
            
            # #拍照
            # ret, frame = cap.read()
            
            #图片
            frame = cv2.imread("D:/Anaconda/envs/Train/Lib/site-packages/ultralytics/assets/road/image.png")
            ret = True
            
            if ret:

                # 获取滑动条值
                Bl = cv2.getTrackbarPos('Bl', WindowName)  
                Gl = cv2.getTrackbarPos('Gl', WindowName)  
                Rl = cv2.getTrackbarPos('Rl', WindowName)  
                Bh = cv2.getTrackbarPos('Bh', WindowName) 
                Gh = cv2.getTrackbarPos('Gh', WindowName)  
                Rh = cv2.getTrackbarPos('Rh', WindowName)  
                ite = cv2.getTrackbarPos('iterations', WindowName)  

                
                #颜色空间转换
                hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
                
                #滤波处理
                hsv_image = cv2.blur(hsv_image,(9,9))

                
                #二值转换，进行颜色分割---》把色域内的像素点设为白色，其余像素点设为黑色
                lower_color = np.array([Bl, Gl, Rl])
                upper_color = np.array([Bh, Gh, Rh])
                mask = cv2.inRange(hsv_image, lower_color, upper_color)
            
                # #开运算
                # hsv_image = cv2.erode(hsv_image, np.ones((3,3),np.uint8), iterations=ite)
                # hsv_image = cv2.dilate(hsv_image, np.ones((3,3),np.uint8), iterations=ite)
                
                
                # #获取色块轮廓（cv2.findContours()函数返回的轮廓列表是按轮廓大小排序的）
                # contours,hierarchy= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)                
                # if contours :
                #     for contour in contours:#筛选出目标色块
                #         x, y, w, h = cv2.boundingRect(contour)
                #         point = cv2.contourArea(contour)
                #         area = w*h
                #         print("面积：",str(area),"色块数量：",str(point),"密度：",str(point/area),"长宽比",str(w/h))
                #         cv2.drawContours(frame,contours,-1,(0,255,0),2)
                
                cv2.namedWindow('mask', cv2.WINDOW_KEEPRATIO)  # 创建一个新窗口
                cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)
                
                cv2.imshow("mask", mask)
                cv2.imshow("frame", frame)
                # 按下 'q' 键退出循环
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break



if __name__ == '__main__':
    main()
    cap.release()# 释放摄像头
    cv2.destroyAllWindows()
