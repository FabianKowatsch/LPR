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

def levenshtein_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _  in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]

def character_error_rate(gt, pred):
    if len(gt) == 0:
        return float('inf')
    return levenshtein_distance(gt, pred) / len(gt)

def word_error_rate(gt, pred):
    gt_words = gt.split()
    pred_words = pred.split()
    if len(gt_words) == 0:
        return float('inf')
    return levenshtein_distance(gt_words, pred_words) / len(gt_words)

def load_images_and_labels(directory_path):
    image_list = []

    # Load images
    for filename in os.listdir(directory_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(directory_path, filename)
            img = cv2.imread(image_path)
            if img is not None:
                # Match the image filename (without extension) to its label
                image_name = os.path.splitext(filename)[0]
                #correct_label = labels.get(image_name, "Unknown")
                image_list.append((img, image_name))

    return image_list

def test(config):
    images = load_images_and_labels(config["data_path"])
    detector = LPD_Module(config["lpd_checkpoint_path"])
    ocr = OCR_Module(config["recognizer"])
    upscaler = Upscaler(config["upscaler"])
    image_processing = Processing(config["image_processing"])

    total_cer = 0
    total_wer = 0
    total_cases = 0
    total_correct = 0

    for i, image_and_label in enumerate(images):
        image = image_and_label[0]
        try:
            boxes = detector(image)
            if not boxes:  # No license plates detected
                continue
        except Exception as e:
            continue
        cers = []
        wers = []
        for j, box in enumerate(boxes):
            try:
                # Crop the image to simplify ocr
                lp_image = crop_image(image, box)

                # Save the cropped image
                cropped_image_path = f'static/uploads/cropped_plate_{i}_{j}.png'
                cv2.imwrite(cropped_image_path, lp_image)

                # Upscaling
                lp_image = upscaler(lp_image)

                # Processing
                lp_image = image_processing(lp_image)

                # Text recognition
                lp_text = ocr(lp_image)

                # Filter text, replace some symbols with spaces
                text_filtered = re.sub(r'[^A-Z0-9]', '', lp_text)

                gt = re.sub(r'[^A-Z0-9]', '', image_and_label[1])
                
                print("Predicted: ", text_filtered)
                print("GT: ", gt)
                print("Correct: ", gt == text_filtered)

                cer = character_error_rate(gt, text_filtered)
                cers.append(cer)
                wer = word_error_rate(gt, text_filtered)
                wers.append(wer)
            
            except Exception as e:
                print(e)
                continue
        

        cer = min(cers)
        wer = min(wers)
        print("CER: ", cer)
        print("WER: ", wer)
        print("")
        total_cer += cer
        total_wer += wer
        total_cases += 1
        if wer == 0:
            total_correct += 1



        
    if total_cases > 0:
        avg_cer = total_cer / total_cases
        avg_wer = total_wer / total_cases
        overall_accuracy = total_correct / total_cases
        print("=== Overall Performance Metrics ===")
        print("Average CER: ", avg_cer)
        print("Average WER: ", avg_wer)
        print("Overall Accuracy: ", overall_accuracy)
    else:
        print("No detections were processed.")

def crop_image(image, box, offset=10):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    # Entferne einen kleinen Rand am linken Rand (offset)
    return image[y1:y2, x1+((x2-x1)//offset):x2, :].copy()

def show_image(img: np.ndarray, plate_image:np.ndarray, text: str, box, ax):

    # draw red outlines
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    image = img.copy()
    image[y1, x1:x2-1, 0] = 255
    image[y2-1, x1:x2-1, 0] = 255
    image[y1:y2-1, x1, 0] = 255
    image[y1:y2-1, x2-1, 0] = 255

    # show image slideshow
    ax[0].clear()
    ax[0].imshow(image)
    ax[0].set_title(f"License plate at: {[x1,x2], [y1,y2]}")
    ax[0].axis('off')

    ax[1].clear() 
    ax[1].imshow(plate_image)
    ax[1].set_title(text)
    ax[1].axis('off')

    plt.draw()
    plt.show()
    #plt.pause(1.0)



def main():
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    test(config)

if __name__ == "__main__":
    main()