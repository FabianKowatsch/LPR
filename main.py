import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import re
from modules.detection import LPD_Module
from modules.ocr import OCR_Module
from modules.upscaling import Upscaler
from modules.image_processing import Processing

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

def load_images_and_labels(directory_path, label_path):
    image_list = []
    labels = {}

    # Extract labels from text file names
    for filename in os.listdir(label_path):
        if filename.endswith('.txt'):
            label = os.path.splitext(filename)[0]  # Remove the .txt extension
            labels[label] = label

    # Load images
    for filename in os.listdir(directory_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(directory_path, filename)
            img = cv2.imread(image_path)
            if img is not None:
                # Match the image filename (without extension) to its label
                image_name = os.path.splitext(filename)[0]
                correct_label = labels.get(image_name, "Unknown")
                image_list.append((img, correct_label))

    return image_list

def test(config):

    images_and_labels = load_images_and_labels(config["data_path"], config["label_path"])
    detector = LPD_Module(config["lpd_checkpoint_path"])
    ocr = OCR_Module(config["recognizer"])
    upscaler = Upscaler(config["upscaler"])
    image_processing = Processing(config["image_processing"])
    visualize = config["visualize"]

    if(visualize):
        plt.ion()
        _, axes = plt.subplots(2, 1)

    for image, correct_label in images_and_labels:

        # LP detection
        boxes = detector(image)

        for box in boxes:

            # Crop the image to simplify ocr
            lp_image = crop_image(image, box)

            # Upscaling
            lp_image = upscaler(lp_image)

            # Processing
            lp_image = image_processing(lp_image)

            # Text recognition
            lp_text = ocr(lp_image)

            # Filter text, replace some symbols with spaces
            text_filtered =  re.sub(r'(?<!\s)[^A-Z0-9-\s](?!\s)', ' ', lp_text)

            if visualize:
                show_image(image, lp_image, text_filtered, correct_label, box, axes)

    if(visualize):      
        plt.ioff()
        #plt.show()

def crop_image(image, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return image[y1:y2, x1:x2, :].copy()

def show_image(img: np.ndarray, plate_image: np.ndarray, predicted_text: str, correct_label: str, box, ax):
    # Draw red outlines
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    image = img.copy()
    image[y1, x1:x2 - 1, 0] = 255
    image[y2 - 1, x1:x2 - 1, 0] = 255
    image[y1:y2 - 1, x1, 0] = 255
    image[y1:y2 - 1, x2 - 1, 0] = 255

    # Show image slideshow
    ax[0].clear()
    ax[0].imshow(image)
    ax[0].set_title(f"License plate at: {[x1, x2], [y1, y2]}")
    ax[0].axis('off')

    ax[1].clear()
    ax[1].imshow(plate_image)
    ax[1].set_title(f"Predicted: {predicted_text} | Correct: {correct_label}")
    ax[1].axis('off')

    plt.draw()
    plt.pause(1.0)



def main():
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    test(config)

if __name__ == "__main__":
    main()