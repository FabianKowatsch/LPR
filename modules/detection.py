import torch
from ultralytics import YOLO
from utils import crop_image, crop_image_xyxy, show_image

class LPD_Module:
    def __init__(self, model_path, verbose=False):
        self.model_detection = YOLO(model_path, verbose=verbose)
        self.model_corners = YOLO("checkpoints/obb/best.pt", False)
 
    def __call__(self, images: torch.Tensor):
        results = self.model_corners(source=images)
        boxes = []
        for result in results:
            #xywhr = result.keypoints.xy  # center-x, center-y, width, height, angle (radians)
            boxes.append(result.obb)
        return boxes
        for result in results:
            for box in result.boxes:
                #print("BOX:", box)
                image = crop_image_xyxy(images, box)
                
                results_corners = self.model_corners(source=image)
                for result_corner in results_corners:
                    print("OBB:", result_corner.obb)
                    try:
                        show_image(image, crop_image(image, result_corner.obb), "", "", box, 0)
                    except Exception as e:
                        continue
                    boxes.append(result_corner.obb)
        return boxes

