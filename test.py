from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov12n.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("E:/ComputerVision_Proj/yolov12/images/test1.jpg", save=True, imgsz=320, conf=0.5)

