import cv2
import numpy as np        
 

'''父子轮廓叠加计算'''
'''前提：二值化完全，cnt中只有目标项存在内外轮廓'''
img = cv2.imread("/home/q/public security/photo/47.jpg")    
frame = img


#理想二值化


# imgBlur2 = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[0, 0, 0])#拓宽黑边
# frame = cv2.copyMakeBorder(frame, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[0, 0, 0])
# gray = cv2.cvtColor(imgBlur2, cv2.COLOR_BGR2GRAY)
# _,imgCanny = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_color = np.array([84, 86, 83])
upper_color = np.array([183, 255, 137])
imgCanny = cv2.inRange(hsv_image, lower_color, upper_color)


cv2.namedWindow("imgCanny", cv2.WINDOW_NORMAL)
cv2.imshow("imgCanny", imgCanny)
cv2.waitKey(0) 


#去墙
a = 0
b = 639
strips = []

for i in range(0,100):
    #拿到一列
    strips = imgCanny[:, i*2]
    strip_array = np.array(strips)
    total_pixels = 480
    white_pixels = np.sum(strip_array == 255)
    #白色占比
    ratio = white_pixels / total_pixels
    #找到墙了
    if(ratio < 0.05):
        a = i*2
        break

strips = []

for i in range(0,100):
    strips = imgCanny[:, 639 - i*2]

    strip_array = np.array(strips)
    total_pixels = 480
    white_pixels = np.sum(strip_array == 255)
    ratio = white_pixels / total_pixels
    if(ratio < 0.05):
        b = 639 - i*2
        break

if(a>b):
    b = a + 10

imgCanny = imgCanny[0:480, a:b]
frame = frame[0:480, a:b]

cv2.namedWindow("imgCanny", cv2.WINDOW_NORMAL)
cv2.imshow("imgCanny", imgCanny)
cv2.waitKey(0) 


#理想二值化

#调试用图
imgCanny_bgr=cv2.cvtColor(imgCanny, cv2.COLOR_BGR2RGB)

contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
hierarchy = hierarchy[0]
print(hierarchy)

c = -1 #计数器
all_approx = 0#总角点
dad_approx = 0#父角点
approx_flag = False#是否需要叠加

for cnt in contours:
    c+=1
    point = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  
    x, y ,w,h = cv2.boundingRect(approx)  
    area = w* h
    
    
    cv2.drawContours(imgCanny_bgr, [cnt], 0, (120, 120, 255), 5)
    cv2.namedWindow("imgCanny_bgr", cv2.WINDOW_NORMAL)
    cv2.imshow("imgCanny_bgr", imgCanny_bgr)
    cv2.waitKey(0)       
    
    #过滤
    if area < 5000 :
        continue
    now_approx = len(approx)
    
    print("目标索引：",c)
    print("本次角点数",now_approx)
    cv2.drawContours(imgCanny_bgr, [cnt], 0, (0, 0, 255), 5)
    cv2.namedWindow("imgCanny_bgr", cv2.WINDOW_NORMAL)
    cv2.imshow("imgCanny_bgr", imgCanny_bgr)
    cv2.waitKey(0)    

    #外内轮廓角点叠加
    if approx_flag:
        all_approx = now_approx + dad_approx
        approx_flag = False
        print("总角点数",all_approx)
    
    #判定下次的数据是否叠加
    if hierarchy[c][2] != -1:
        #记录此次的角点数
        dad_approx = now_approx
        print("dad_approx",dad_approx)
        approx_flag = True
    

    # #形状判断
    # approx = np.array(approx)
    # approx = np.squeeze(approx, axis=1)
    # sorted_points = approx[np.argsort(approx[:, 1],)] #按y值从小到大排序

    #------->x
    #|   A   B
    #|   C   D
    #v
    
    # #AB
    # if sorted_points[0][0] <= sorted_points[1][0]:
    #     A = sorted_points[0]
    #     B = sorted_points[1]
    # elif sorted_points[0][0] > sorted_points[1][0]:
    #     A = sorted_points[1]
    #     B = sorted_points[0]
    # #CD
    # if sorted_points[2][0] <= sorted_points[3][0]:
    #     C = sorted_points[2]
    #     D = sorted_points[3]
    # elif sorted_points[2][0] >= sorted_points[3][0]:
    #     C = sorted_points[3]
    #     D = sorted_points[2]

    
    #调试：以A,B,C,D为圆心画圆形    
    # cv2.circle(imgCanny_bgr, (A[0], A[1]), 5, (0, 0, 255), -1)
    # cv2.circle(imgCanny_bgr, (B[0], B[1]), 5, (0, 0, 255), -1)
    # cv2.circle(imgCanny_bgr, (C[0], C[1]), 5, (0, 0, 255), -1)
    # cv2.circle(imgCanny_bgr, (D[0], D[1]), 5, (0, 0, 255), -1)
