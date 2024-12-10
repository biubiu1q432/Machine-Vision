import os
import time
import cv2

#读取视频
cap = cv2.VideoCapture('D:\Anaconda\envs\YOLO\TRAIN_ROAD\data\part-2.mp4')
cnt = 0
#播放视频
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('frame', frame)
        cnt+=1
        #按照cnt动态存照片
        if cnt%20==0:
            cv2.imwrite('D:\Anaconda\envs\YOLO\TRAIN_ROAD\data\part3_%d.jpg' %cnt, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    else:
        break



