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
from utils import crop_image, normalize_text

def load_images(path):
    image_list = []
    
    # If path is a single file
    if os.path.isfile(path):
        if path.endswith(('.jpg', '.png', '.jpeg')):  # Add support for PNG and other formats
            img = cv2.imread(path)
            if img is not None:
                image_list.append(img)
        else:
            raise ValueError(f"Unsupported file format: {path}")
    
    # If path is a directory
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(path, filename)
                img = cv2.imread(image_path)
                if img is not None:
                    image_list.append(img)
    
    else:
        raise FileNotFoundError(f"Path not found: {path}")
    
    if not image_list:
        raise ValueError("No valid images found in the specified path.")
    
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

def predict(config):
    images = load_images(config["data_path"])
    detector = LPD_Module(config["lpd_checkpoint_path"], config["verbose"])
    ocr = OCR_Module(config["recognizer"])
    upscaler = Upscaler(config["upscaler"])
    image_processing = Processing(config["image_processing"])
    visualize = config["visualize"]
    
    results = []  # List to store results

    for i, image in enumerate(images):
        try:
            boxes = detector(image)
            if not boxes:  # No license plates detected
                results.append({
                    "image": None,
                    "error": "No license plates detected."
                })
                continue
        except Exception as e:
            results.append({
                "image": None,
                "error": f"License plate detection failed: {str(e)}"
            })
            continue

        for j, box in enumerate(boxes):
            try:
                # Crop the image to simplify ocr
                lp_image = crop_image(image, box)


                # Upscaling
                lp_image = upscaler(lp_image)

                # Processing
                lp_image = image_processing(lp_image, box, upscaler.get_scale())


                # Save the cropped image
                cropped_image_path = f'static/uploads/cropped_plate_{i}_{j}.png'
                cv2.imwrite(cropped_image_path, lp_image)


                # Text recognition
                lp_text = ocr(lp_image)

                lp_text_normalized = normalize_text(lp_text)
                # Filter text, replace some symbols with spaces
                #text_filtered =  re.sub(r'(?<!\s)[^A-Z0-9-\s](?!\s)', ' ', lp_text)
                text_filtered = re.sub(r'[^A-Z0-9]', '', lp_text_normalized)
                text_filtered = re.sub(r'([A-Z0-9])\1{4}', lambda m: m.group(1) * 4, text_filtered)

                box_serializable = box.xyxyxyxy.flatten().cpu().numpy().tolist() if hasattr(box, 'xyxyxyxy') else str(box)

                # If OCR failed, add only the error and skip the filtering
                if "OCR failed" in lp_text:
                    results.append({
                        "image": cropped_image_path,
                        "box": box_serializable,
                        "lp_text": lp_text,
                        "error": "OCR processing failed: No valid text detected."
                    })
                else:
                    # Filter text and append a valid result
                    #text_filtered = re.sub(r'(?<!\s)[^A-Z0-9-\s](?!\s)', ' ', lp_text)

                    results.append({
                        "image": cropped_image_path,
                        "box": box_serializable,
                        "lp_text": lp_text,
                        "text_filtered": text_filtered
                    })
            except Exception as e:
                box_serializable = box.xyxyxyxy.flatten().cpu().numpy().tolist() if hasattr(box, 'xyxyxyxy') else str(box)
                print(e)
                results.append({
                    "image": cropped_image_path,
                    "box": box_serializable,
                    "error": f"OCR processing failed: {str(e)}"
                })
                continue

    return results


def predict_from_video(config, progress_callback=None):
    video_path = config["data_path"]
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detector = LPD_Module(config["lpd_checkpoint_path"], verbose=config["verbose"])
    ocr = OCR_Module(config["recognizer"])
    upscaler = Upscaler(config["upscaler"])
    image_processing = Processing(config["image_processing"])
    frame_interval = config["frame_interval"]
    
    frame_count = 0
    results = []

    # Create output folder if it doesn't exist
    cropped_dir = "static/uploads/"
    os.makedirs(cropped_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)  # Get video FPS

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when the video ends

        if frame_count % frame_interval == 0:
            if progress_callback:
        # Using processed frames over total frames (approximate)
                progress = int((frame_count / total_frames) * 100)
                progress_callback(progress)
            boxes = detector(frame)
            for j, box in enumerate(boxes):
                try:
                    lp_image = crop_image(frame, box)

                    # Save the cropped license plate image
                    cropped_image_path = f"{cropped_dir}cropped_plate_frame_{frame_count}_plate_{j}.png"
                    cv2.imwrite(cropped_image_path, lp_image)

                    # Upscale
                    lp_image = upscaler(lp_image)

                    # Process
                    lp_image = image_processing(lp_image, box)

                    # OCR
                    lp_text = ocr(lp_image)
                    text_filtered = re.sub(r'(?<!\s)[^A-Z0-9-\s](?!\s)', ' ', lp_text)

                    if "OCR failed" in lp_text:
                        continue

                    # Calculate timestamp (HH:MM:SS)
                    seconds = frame_count / fps
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    seconds = int(seconds % 60)
                    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                    box_serializable = box.xyxyxyxy.flatten().cpu().numpy().tolist() if hasattr(box, 'xyxyxyxy') else str(box)

                    # Append results with cropped image
                    results.append({
                        "frame": frame_count,
                        "fps": fps,
                        "time": time_str,
                        "image": cropped_image_path,  # Add the path to the cropped image
                        "box": box_serializable,
                        "lp_text": lp_text,
                        "text_filtered": text_filtered
                    })

                except Exception as e:
                    print(f"Error processing plate in frame {frame_count}: {e}")
                    continue

        frame_count += 1

    cap.release()
    print(f"First Box: {results[0]['box']}")
    return results

def main():
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # test(config)
    config['data_path'] = '/mnt/c/Users/jovab/Desktop/Licence_Plate_Camera_Illustration_Video.mkv'
    config['frame_interval'] = 5
    results = predict_from_video(config)
    print(results)

if __name__ == "__main__":
    main()