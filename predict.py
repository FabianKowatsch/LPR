import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import re

from deep_sort_realtime.deepsort_tracker import DeepSort
from modules.detection import LPD_Module
from modules.ocr import OCR_Module
from modules.upscaling import Upscaler
from modules.image_processing import Processing
from utils.utils import crop_image, show_image, normalize_text, calculate_iou

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

def test(config):

    images_and_labels = load_images_and_labels(config["data_path"], config["label_path"])
    detector = LPD_Module(config["lpd_checkpoint_path"], config["verbose"])
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
            xyxy = map(int, box.xyxy[0])
            lp_image = crop_image(image, xyxy)

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
                xyxy = map(int, box.xyxy[0])
                lp_image = crop_image(image, xyxy)


                # Upscaling
                lp_image = upscaler(lp_image)

                # Processing
                lp_image = image_processing(lp_image)


                # Save the cropped image
                cropped_image_path = f'static/uploads/cropped_plate_{i}_{j}.png'
                cv2.imwrite(cropped_image_path, lp_image)


                # Text recognition
                lp_text, confidence = ocr(lp_image)

                # Filter text, replace some symbols with spaces
                lp_text_normalized = normalize_text(lp_text)
                text_filtered = re.sub(r'[^A-Z0-9]', '', lp_text_normalized)
                text_filtered = re.sub(r'([A-Z0-9])\1{4}', lambda m: m.group(1) * 4, text_filtered)

                box_serializable = box.xyxy.cpu().numpy().tolist() if hasattr(box, 'xyxy') else str(box)

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
                    text_filtered = re.sub(r'(?<!\s)[^A-Z0-9-\s](?!\s)', ' ', lp_text)

                    results.append({
                        "image": cropped_image_path,
                        "box": box_serializable,
                        "lp_text": lp_text,
                        "text_filtered": text_filtered
                    })
            except Exception as e:
                box_serializable = box.xyxy.cpu().numpy().tolist() if hasattr(box, 'xyxy') else str(box)

                if(len(results) == 0):
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

    # Create output folder if it doesn't exist
    cropped_dir = "static/uploads/"
    os.makedirs(cropped_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)  # Get video FPS

    # Initialize DeepSORT tracker
    tracker = DeepSort(
        max_age=20,           # How long an object is tracked without detections
        # n_init=2,           # Minimum number of confirmed detections before being tracked
        # max_cosine_distance=0.3,  # Cosine distance threshold for feature matching
    )
    

    frame_count = 0
    max_track_id = -1
    results = []
    while cap.isOpened():
        # print(f"___Frame {frame_count}___")
        ret, frame = cap.read()
        if not ret:
            break  # Stop when the video ends

        # Calculate timestamp (HH:MM:SS)
        seconds = frame_count / fps
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        print(f"Time: {time_str}")

        if frame_count % frame_interval == 0:
            if progress_callback:
                # Using processed frames over total frames (approximate)
                progress = int((frame_count / total_frames) * 100)
                progress_callback(progress)

            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0  # Shape: [3, H, W], normalized
            # frame_tensor = frame_tensor.unsqueeze(0).to(detector.model.device)  # Add batch dim and move to device

            # print(f"Frame data type: {type(frame)}")
            boxes = detector(frame)
            center_boxes = []  # Prepare for DeepSORT
            bounding_boxes = []


            for j, box in enumerate(boxes):
                try:
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy().tolist()[0]

                    # For Debugging
                    results.append({
                        "frame": frame_count,
                        "fps": fps,
                        "box_raw": [x1, y1, x2, y2]
                    })

                    w = x2 - x1
                    h = y2 - y1
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    conf = 1.0  # Assume high confidence since we trust the detector
                    
                    center_boxes.append(([cx, cy, w, h], conf, "license_plate"))  # Format for DeepSORT
                    bounding_boxes.append([x1, y1, x2, y2])

                except Exception as e:
                    print(f"Error processing plate in frame {frame_count}: {e}")
                    continue

            # Update tracker with center_boxes
            # tracked_objects = last_frame_tracked_objects
            # last_frame_tracked_objects = tracker.update_tracks(center_boxes, frame=frame)
            tracked_objects = tracker.update_tracks(center_boxes, frame=frame)
            
            # tracked_boxes = []
            for track in tracked_objects:
                if not track.is_confirmed():
                    print("Not confirmed")
                    continue
                
                track_id = track.track_id  # Unique object ID from DeepSORT
                if int(track_id) > max_track_id:
                    max_track_id = int(track_id)

                tracker_box = track.to_tlbr(orig=True)

                # Check if the tracker box overlaps with any of the bounding boxes
                detector_box = None
                iou_max = 0
                for box in bounding_boxes:
                    iou = calculate_iou(box, tracker_box)
                    if iou > iou_max:
                        iou_max = iou
                        detector_box = box
                        # tracked_boxes.append(box)

                if detector_box is None:
                    continue
                                    
                # Use YOLO bounding box (not DeepSORT's predicted one)
                x1, y1, x2, y2 = list(map(int, detector_box))
                try:
                    # Crop and process image
                    lp_image = crop_image(frame, list(map(int, detector_box)))

                    # Save cropped image
                    cropped_image_path = f"{cropped_dir}tracked_plate_{track_id}_frame_{frame_count}.png"
                    cv2.imwrite(cropped_image_path, lp_image)

                    # Upscale and process
                    lp_image = upscaler(lp_image)
                    lp_image = image_processing(lp_image)

                    # OCR
                    lp_text, confidence = ocr(lp_image)
                    text_filtered = re.sub(r'(?<!\s)[^A-Z0-9-\s](?!\s)', ' ', lp_text)

                    if "OCR failed" in lp_text:
                        continue


                    # Append results with tracking ID
                    results.append({
                        "is_tracked": True,
                        "frame": frame_count,
                        "fps": fps,
                        "time": time_str,
                        "track_id": int(track_id),  # Add tracking ID
                        "image": cropped_image_path,  # Path to cropped image
                        "confidence": confidence,
                        "box": [x1, y1, x2, y2],
                        "lp_text": lp_text,
                        "text_filtered": text_filtered
                    })

                except Exception as e:
                    print(f"Error processing tracked plate {track_id} in frame {frame_count}: {e}")
                    continue
        frame_count += 1

    cap.release()
    if (len(results) == 0):
        print("No license plates detected in the video.")
    else:
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