import torch
import numpy as np
from ultralytics import YOLO
import torchvision.transforms.functional as TF

class LPD_Module:
    def __init__(self, model_path, verbose=False):
        self.model = YOLO(model_path, verbose=verbose)

        if torch.cuda.is_available():
            self.model.to('cuda')
        else:
            self.model.to('cpu')
 
    def __call__(self, images):
        # images_tensor = self.prepare_image(images)
        results = self.model(source=images, device=self.model.device)
        boxes = []
        for result in results:
            for box in result.boxes:
                #plate_box = self.crop_image_tensor(images, box)
                boxes.append(box)
        return boxes
    
    def prepare_image(self, images):
        if type(images) == np.ndarray:
            tensor = torch.from_numpy(images).float().to(self.model.device)
        elif type(images) == torch.Tensor:
            tensor = images

        # Convert HWC -> CHW
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)

        # Normalize to [0, 1] if values are 0–255
        if tensor.max() > 1:
            tensor = tensor / 255.0

        # Resize to a shape divisible by 32, e.g. 640x640
        tensor = TF.resize(tensor, [640, 640])  # use torchvision or your own resizing

        # Add batch dimension: CHW -> BCHW
        tensor = tensor.unsqueeze(0)

        return tensor

