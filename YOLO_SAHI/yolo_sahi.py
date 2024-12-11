import cv2
from sahi import AutoDetectionModel
from tkinter import *
from sahi.predict import get_sliced_prediction
from sahi.predict import get_prediction


#普通预测
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=r"C:\Users\86135\Desktop\Machine_Vision\YOLO_SAHI\models\best_train.pt",
    confidence_threshold=0.3,
    device="cuda:0",  # or 'cuda:0'
    image_size=640,

)

result_noslice = get_prediction(r"C:\Users\86135\Desktop\Machine_Vision\TRAIN_ROAD\data\part2_1000.jpg", detection_model)
result_noslice.export_visuals(export_dir=r"demo_data/")
cv2.imshow("image", cv2.imread(r"demo_data/prediction_visual.png"))
cv2.waitKey(0)



#切片预测
result_slice = get_sliced_prediction(
    r"C:\Users\86135\Desktop\Machine_Vision\TRAIN_ROAD\data\part2_1000.jpg",
    detection_model,
    slice_height=160,
    slice_width=160,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

result_slice.export_visuals(export_dir="demo_data/",file_name="prediction_visual_slice")
cv2.imshow("img", cv2.imread(r"demo_data/prediction_visual_slice.png"))
cv2.waitKey(0)


num_sliced_dets = len(result_slice.object_prediction_list)
num_orig_dets = len(result_noslice.object_prediction_list)

print(f"Detections predicted without slicing: {num_orig_dets}")
print(f"Detections predicted with slicing: {num_sliced_dets}")