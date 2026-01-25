from ultralytics import YOLO
import torch
def model(self):
   model = YOLO("yolov8n.pt")
   model.train(data="/dataset",
               epochs=5,

                               
                             
                             
                             )