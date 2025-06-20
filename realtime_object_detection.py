"""
names: {0: 'prson', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 
17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 
24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 
42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 
49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 
56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 
63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 
77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
"""

import cv2
from ultralytics import YOLO
import time

# model = YOLO("yolov12n.pt")
model = YOLO("yolov12s.pt")

# 打开视频文件
# cap = cv2.VideoCapture("path.mp4")

# 或使用设备“0”打开视频捕获设备读取帧
cap = cv2.VideoCapture(1)

# 设置视频帧大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

title = "YOLOv12 RealTime_Inference"
# 设置窗口位置
cv2.namedWindow(title, cv2.WINDOW_NORMAL)
cv2.moveWindow(title, 200, 100)

# 循环播放视频帧
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()
    if success:
        # 在框架上运行 YOLOv8 推理
        results = model(frame,classes=[0, 2], conf=0.5)
        # print(results)
        class_ids = results[0].boxes.cls.cpu().numpy()
        labels = [model.names[int(cls)] for cls in class_ids]
        labels_set = set(labels)
        
        # 设置感兴趣的类别并进行保存
        
        check_flag = ('person' in labels_set or 'car' in labels_set)
        
        if check_flag:
            # （选择1）保留原始图像
            # save_img = results[0].orig_img
            # （选择2）保存分割图像
            save_img = results[0].plot()
            # 年月日时分秒进行命名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # 保存图像
            save_img_path = f"./save_images/{timestamp}.jpg"
            cv2.imwrite(save_img_path, save_img)
        # break
        # 在框架上可视化结果
        annotated_frame = results[0].plot()
        # 显示带标注的框架
        cv2.imshow(title, annotated_frame)
        # 如果按下“q”，则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果到达视频末尾，则中断循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()
