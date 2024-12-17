import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from modules.detection import LPD_Module
from modules.ocr import OCR_Module
import yaml

def load_images(directory_path):
    image_list = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(directory_path, filename)
            img = cv2.imread(image_path)
            if img is not None:
                image_list.append(img)
    #images = torch.tensor(np.stack(image_list, axis=0))
    #images = images.permute(0, 3, 1, 2)
    return image_list

def test(data_path: str, detector: LPD_Module, ocr: OCR_Module, visualize=True):
    images = load_images(data_path)

    if(visualize):
        plt.ion()
        _, axes = plt.subplots()

    for image in images:

        # LP detection
        boxes = detector.detect(image)

        for box in boxes:

            # Crop the image to simplify ocr
            lp_image = crop_image(image, box)

            # Text recognition
            lp_text = ocr.recognize(lp_image)

            if(visualize):
                show_image(image, lp_text, box, axes)

    if(visualize):      
        plt.ioff()
        #plt.show()

def crop_image(image, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return image[y1:y2, x1:x2, :]

def show_image(image: np.ndarray, text: str, box, ax):

    # draw red outlines
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    image[y1, x1:x2-1, 0] = 255
    image[y2-1, x1:x2-1, 0] = 255
    image[y1:y2-1, x1, 0] = 255
    image[y1:y2-1, x2-1, 0] = 255

    # show image slideshow
    ax.clear()
    ax.imshow(image)
    ax.set_title(f"Detected Text: {text}")
    ax.axis('off')
    plt.draw()
    plt.pause(1.0)



def main():
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    detector = LPD_Module(config["lpd_checkpoint_path"])
    ocr = OCR_Module(config)

    test(config["data_path"], detector, ocr)

if __name__ == "__main__":
    main()