import os
import cv2
from modules.detection import LPD_Module

def train():
    detector = LPD_Module("checkpoints/pretrained/yolo11n.pt")
    #results = detector.model.train(data="config/data.yaml", epochs=400, imgsz=640, device=0, val=True, single_cls=True, batch=32)
    results = detector.model.train(cfg="config/train.yaml")
    #print(results)
if __name__ == "__main__":
    train()