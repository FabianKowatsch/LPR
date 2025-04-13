import torch
from ultralytics import YOLO

class LPD_Module:
    def __init__(self, model_path, verbose=False):
        self.model = YOLO(model_path, verbose=verbose)

        if torch.cuda.is_available():
            self.model.to('cuda')
        else:
            self.model.to('cpu')
 
    def __call__(self, images):
        results = self.model(source=images, device=self.model.device)
        boxes = []
        for result in results:
            for box in result.boxes:
                #plate_box = self.crop_image_tensor(images, box)
                boxes.append(box)
        return boxes

