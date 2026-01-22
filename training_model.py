from ultralytics import YOLO

def model(self):
   model = YOLO("yolov8n.pt")
   model.train(data="/dataset")