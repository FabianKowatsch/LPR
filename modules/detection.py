import torch
from ultralytics import YOLO

class LPD_Module:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, images: torch.Tensor):
        results = self.model(source=images)
        boxes = []
        for result in results:
            for box in result.boxes:
                #plate_box = self.crop_image_tensor(images, box)
                boxes.append(box)
        return boxes
