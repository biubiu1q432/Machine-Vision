import cv2
import numpy as np

#预处理
path=r"C:\Users\86135\Desktop\Machine_Vision\Shape_fitting\point3.jpg"
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, 0)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(thresh,kernel,iterations = 1)


judge_index = 2 #分界线
y_dis = 43  #步长
x_dis = 5

points = []
colors = []
center_points=[]
p1 = []
p2 = []
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for contour in contours:
    x_, y_, w_, h_ = cv2.boundingRect(contour)

    y_down = y_+h_ - 5
    y_up = y_+5
    x_left = x_ - 20
    x_right = x_+w_+ 20

    for y in range(y_down,y_up,-y_dis):        
        for x in range(x_left,x_right,x_dis):
            points.append((x,y))
            colors.append(dilation[y,x])
         
            #如果dilation后三个数都是255，前面所有数都是0
            if colors[-judge_index:].count(255) == judge_index and colors[:-judge_index].count(0) == len(colors)-judge_index and len(colors)> judge_index:
                cv2.circle(img, (points[-judge_index][0], points[-judge_index][1]), 5, (0, 0, 255), -1)
                p1 = points[-judge_index]
                points = []
                colors = []

            #如果dilation后三个数都是0，前面所有数都是255
            if colors[-judge_index:].count(0) == judge_index and colors[:-judge_index].count(255) == len(colors)-judge_index and len(colors) > judge_index:
                cv2.circle(img, (points[-judge_index][0], points[-judge_index][1]), 5, (0, 255, 0), -1)
                p2 = points[-judge_index]
                points = []
                colors = []

            #取中点
            if p1 != [] and p2 != []:
                center_point = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
                cv2.circle(img, (center_point[0], center_point[1]), 15, (255, 0, 255), -1)
                center_points.append(center_point)
                p1 = []
                p2 = []

    print(center_points)
    cv2.imshow('img', img)
    cv2.waitKey(0)
