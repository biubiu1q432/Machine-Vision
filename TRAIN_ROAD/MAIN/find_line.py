import cv2
import numpy as np
from PIL import Image
import os


img = cv2.imread(r'D:\Anaconda\envs\YOLO\TRAIN_ROAD\point3 copy.jpg')

'''二值化'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


'''取中点'''
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour=contours[2]
#矩形框
x,y, w, h = cv2.boundingRect(contour)


#上
h1 = y +10
cv2.line(img, (0, h1), (x+w, h1), (900, 255, 0), 2)
cv2.imshow("img", img)
cv2.waitKey(0)
x_min = 10000000
x_max = 0
for c in contour:
    y_=c[0][1]
    x_=c[0][0]
    #匹配成功，找左右x
    if y_ == h1:
        if x_ < x_min:
            x_min = x_
        if x_ > x_max:
            x_max = x_

left_up = (x_min, h1)
right_up = (x_max, h1)

print(x_min, x_max)
cv2.circle(img, (x_min, h1), 15, (0, 0, 255), -1)
cv2.circle(img, (x_max, h1), 15, (0, 0, 255), -1)
cv2.imshow("img", img)
cv2.waitKey(0)


#中
h = y+h//2
cv2.line(img, (0, h), (x+w, h), (900, 255, 0), 2)
cv2.imshow("img", img)
cv2.waitKey(0)
x_min = 10000000
x_max = 0
for c in contour:
    y_=c[0][1]
    x_=c[0][0]
    #匹配成功，找左右x
    if y_ == h:
        if x_ < x_min:
            x_min = x_
        if x_ > x_max:
            x_max = x_

left_mid = (x_min, h)
right_mid = (x_max, h)
print(x_min, x_max)
cv2.circle(img, (x_min, h), 15, (0, 0, 255), -1)
cv2.circle(img, (x_max, h), 15, (0, 0, 255), -1)
cv2.imshow("img", img)
cv2.waitKey(0)





#下
h2 = y+h-10
cv2.line(img, (0, h2), (x+w, h2), (900, 255, 0), 2)
cv2.imshow("img", img)
cv2.waitKey(0)
x_min = 10000000
x_max = 0
for c in contour:
    y_=c[0][1]
    x_=c[0][0]
    #匹配成功，找左右x
    if y_ == h2:
        if x_ < x_min:
            x_min = x_
        if x_ > x_max:
            x_max = x_

left_down = (x_min, h2)
right_down = (x_max, h2)
print(x_min, x_max)
cv2.circle(img, (x_min, h2), 15, (0, 0, 255), -1)
cv2.circle(img, (x_max, h2), 15, (0, 0, 255), -1)

cv2.imshow("img", img)
cv2.waitKey(0)

#画线
cv2.line(img, left_up, right_down, (0, 255, 0), 2)
cv2.line(img, left_down, right_up, (0, 255, 0), 2)
cv2.imshow("img", img)
cv2.waitKey(0)



# #取得矩形框四个定点并画圆形
# left_up = (x, y)
# right_down = (x + w, y + h)
# left_down = (x, y + h)
# right_up = (x + w, y)
# cv2.circle(img, left_up, 15, (0, 0, 255), -1)
# cv2.circle(img, right_down, 15, (0, 0, 255), -1)
# cv2.circle(img, left_down, 15, (0, 0, 255), -1)
# cv2.circle(img, right_up, 15, (0, 0, 255), -1)
# cv2.imshow("img", img)
# cv2.waitKey(0)

# #画线
# cv2.line(img, left_up, right_down, (0, 255, 0), 2)
# cv2.line(img, left_down, right_up, (0, 255, 0), 2)
# cv2.imshow("img", img)
# cv2.waitKey(0)



# #从矩形框下侧向上遍历
# for now_y in range( y,y-h,-20):
#     #寻找y值与i
#     for c in contour:
#         c_y=c[0][1]
#         c_x=c[0][0]
#         if c_y == now_y:
#             cv2.circle(img, (c_x, c_y),15, (0, 0, 255), -1)
#             cv2.imshow("img", img)
#             cv2.waitKey(0)

