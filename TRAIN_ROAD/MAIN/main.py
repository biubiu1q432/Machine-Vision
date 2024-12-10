import json
import numpy as np
from ultralytics import YOLO
import cv2

source_video_path =r"data\part-2.mp4"
train_model_path = r"my_model\best_train.pt"
road_model_path=r"my_model\best_road.pt"
train_model = YOLO(train_model_path)
road_model= YOLO(road_model_path)


def find_line(img):
    #二值化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 轮廓检测
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    contour=contours[2]
    x,y, w, h = cv2.boundingRect(contour)
    h_up = y +10
    cv2.line(img, (0, h_up), (x+w, h_up), (900, 255, 0), 2)
    cv2.imshow("img", img)
    cv2.waitKey(200)
    x_min = 10000000
    x_max = 0
    for c in contour:
        y_=c[0][1]
        x_=c[0][0]
        #匹配成功，找左右x
        if y_ == h_up:
            if x_ < x_min:
                x_min = x_
            if x_ > x_max:
                x_max = x_

    left_up = np.array([x_min, h_up])
    right_up = np.array([x_max, h_up])
    cv2.circle(img, (x_min, h_up), 15, (0, 0, 255), -1)
    cv2.circle(img, (x_max, h_up), 15, (0, 0, 255), -1)
    cv2.imshow("img", img)
    cv2.waitKey(200)

    #下
    h_down = y+h-10
    cv2.line(img, (0, h_down), (x+w, h_down), (900, 255, 0), 2)
    cv2.imshow("img", img)
    cv2.waitKey(200)
    x_min = 10000000
    x_max = 0
    for c in contour:
        y_=c[0][1]
        x_=c[0][0]
        #匹配成功，找左右x
        if y_ == h_down:
            if x_ < x_min:
                x_min = x_
            if x_ > x_max:
                x_max = x_

    left_down = np.array([x_min, h_down])
    right_down = np.array([x_max, h_down])
    print(x_min, x_max)
    cv2.circle(img, (x_min, h_down), 15, (0, 0, 255), -1)
    cv2.circle(img, (x_max, h_down), 15, (0, 0, 255), -1)

    cv2.imshow("img", img)
    cv2.waitKey(200)

    #画线
    cv2.line(img, left_up, right_down, (0, 255, 0), 2)
    cv2.line(img, left_down, right_up, (0, 255, 0), 2)
    cv2.imshow("img", img)
    cv2.waitKey(200)

    return left_up, right_down, left_down, right_up

def get_distance_from_point_to_line(point, line_point1, line_point2):
    #对于两点坐标为同一点时,返回点与点的距离
    if (line_point1 == line_point2).all():
        point_array = np.array(point)
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array -point1_array )
    #计算直线的三个参数
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    #根据点到直线的距离公式计算距离
    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
    return distance


'''轨道检测'''
cap = cv2.VideoCapture(source_video_path)
for i in range(0,20):
    success, frame = cap.read() 
road_results = road_model(frame,show=True,conf=0.95)

#获取掩码边缘坐标
mask = np.zeros_like(frame)
for result in road_results:
    masks = result.masks         
masks_xy = masks.xy    
masks_xy = [np.array(mask).astype(int) for mask in masks_xy]
#把在掩码内的像素点设为白色
for mask_xy in masks_xy:
    cv2.fillPoly(mask, [mask_xy], (255, 255, 255))
#找轨道
reconstructed_image = mask
left_up, right_down, left_down, right_up=find_line(reconstructed_image)
'''轨道检测'''


'''火车在轨检测'''
cap = cv2.VideoCapture(source_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output_video.avi', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    success, frame = cap.read()        
    data=[]
    last_data=[]
    if success:
        #追踪火车
        train_results = train_model.track(frame, persist=True,tracker="bytetrack.yaml",conf=0.5,verbose=False)        
        #获取火车中心坐标
        for box in train_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if box.id != None:
                id = int(box.id.item()) 
            else:
                id = 0
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            #把center_x和center_y以坐标的形式放入train_xy
            train_xy = np.array([center_x,center_y])

            #画出火车中心
            cv2.circle(frame, (center_x, center_y), 10, (255, 255, 255), -1)

            #点线距离
            dis1 = get_distance_from_point_to_line(train_xy,left_up, right_down)
            dis2 = get_distance_from_point_to_line(train_xy,left_down, right_up)

            #判定
            if abs(dis1-dis2) <=30:
                green_track = "occupied"
                blue_track = "occupied"
            elif dis1 > dis2:
                green_track = "occupied"
                blue_track = "empty"
            else:
                green_track = "empty"
                blue_track = "occupied"
            new_data=  {
                "frame_id": id,
                "track_status": {
                    "green_track": green_track,
                    "blue_track": blue_track
                }
                }
            data.append(new_data)
        
        if data != []:
            json_data = json.dumps(data, indent=4)
            print(json_data)

        # 展示带注释的帧
        frame = train_results[0].plot()
        #绘制直线一次
        cv2.line(frame, left_up, right_down, (255, 255, 0), 5)
        cv2.line(frame, left_down, right_up, (0, 255, 0), 5)   
        cv2.imshow("YOLOv8 Tracking", frame)
        # # 写入帧到视频文件
        # out.write(frame)  
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
'''火车在轨检测'''


cap.release()
cv2.destroyAllWindows()