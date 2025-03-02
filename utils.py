import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

def get_min_max_values(box):
    x1, y1, x2, y2, x3, y3, x4, y4 = map(int, box.xyxyxyxy[0].flatten())
    ys = [y1, y2, y3, y4]
    xs = [x1, x2, x3, x4]
    return  np.min(xs),  np.max(xs),  np.min(ys),  np.max(ys)

def crop_image(image, box, offset_left=40, offset_right=30):
    x1, x2, y1, y2 = get_min_max_values(box)
    #return image[y1:y2+ ((y2-y1) // 5) , x1:x2].copy()
    return image[y1:y2 + ((y2-y1) // 5), (x1 + ((x2-x1)//offset_left)): (x2 - ((x2-x1)//offset_right))].copy()


    #return image[y1:y2+ ((y2-y1) // 5) , x1:x2].copy()
    #return image[y1:y2 + ((y2-y1) // 5), (x1 + ((x2-x1)//offset_left)): (x2 - ((x2-x1)//offset_right))].copy()


def crop_image_xyxy(image, box, offset_left=13, offset_right=30):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return image[y1:y2, (x1 + ((x2-x1)//offset_left)) : (x2 - ((x2-x1)//offset_right)), :].copy()


def show_image(img: np.ndarray, plate_image: np.ndarray, predicted_text: str, correct_label: str, box, ax):
    # Draw red outlines
    x1, x2, y1, y2 = get_min_max_values(box)
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

def sort_rectangle_points(pts):

    sum_coords = pts.sum(axis=1)
    diff_coords = np.diff(pts, axis=1)

    top_left = pts[np.argmin(sum_coords)]

    bottom_right = pts[np.argmax(sum_coords)]

    top_right = pts[np.argmin(diff_coords)]

    bottom_left = pts[np.argmax(diff_coords)]

    sorted_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    return sorted_pts

def get_skewed_rectangle_dimensions(sorted_points):
    width_top = euclidean_distance(sorted_points[0], sorted_points[1])   # Top side
    width_bottom = euclidean_distance(sorted_points[3], sorted_points[2]) # Bottom side
    height_left = euclidean_distance(sorted_points[0], sorted_points[3])  # Left side
    height_right = euclidean_distance(sorted_points[1], sorted_points[2]) # Right side

    # Average width and height
    avg_width = int(width_top + width_bottom) // 2
    avg_height = int(height_left + height_right) // 2
    return avg_width, avg_height

def euclidean_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def normalize_text(text):
    replacements = {
        'Ä': 'A', 'Ö': 'O', 'Ü': 'U',
        'ä': 'A', 'ö': 'O', 'ü': 'U',
        'ß': 'SS'
    }
    for umlaut, replacement in replacements.items():
        text = text.replace(umlaut, replacement)
    return text

def create_color_mask(image, threshold=80, ratio_red=1.6, ratio_green=1.2, ratio_blue=1.3):
    red = image[:, :, 2]
    green = image[:, :, 1]
    blue = image[:, :, 0]

    # "Too Red" mask
    too_red_mask = (red > threshold) & (red > green * ratio_red) & (red > blue * ratio_red)

    # "Too Green" mask
    too_green_mask = (green > threshold) & (green > red * ratio_green) & (green > blue * ratio_green)

    # "Too Blue" mask
    too_blue_mask = (blue > threshold) & (blue > red * ratio_blue) & (blue > green * ratio_blue)

    return too_red_mask, too_green_mask, too_blue_mask

def calculate_char_confusion_matrix(gt_words, pred_words):
    # Step 1: Flatten words into characters
    gt_chars = ''.join(gt_words)  # join the ground truth words into a single string of characters
    pred_chars = ''.join(pred_words)  # join the predicted words into a single string of characters

    # Step 2: Define the set of unique characters that might appear
    unique_chars = set(gt_chars + pred_chars)  # include all characters from both ground truth and prediction

    # Step 3: Initialize a confusion matrix (as a defaultdict)
    confusion_matrix = defaultdict(lambda: defaultdict(int))  # stores counts of character pairs

    # Step 4: Compare corresponding characters in ground truth and prediction
    for g, p in zip(gt_chars, pred_chars):
        confusion_matrix[g][p] += 1

    # Step 5: Convert the confusion matrix into a regular NumPy matrix for easier visualization
    all_chars = sorted(unique_chars)  # list of all unique characters sorted
    char_index = {char: idx for idx, char in enumerate(all_chars)}  # map characters to indices
    
    # Create an empty confusion matrix
    matrix = np.zeros((len(all_chars), len(all_chars)), dtype=int)
    
    # Fill the confusion matrix with values from the defaultdict
    for g in confusion_matrix:
        for p in confusion_matrix[g]:
            matrix[char_index[g], char_index[p]] = confusion_matrix[g][p]

    return matrix, all_chars


def plot_confusion_matrix(matrix, all_chars):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create a heatmap using imshow
    cax = ax.matshow(matrix, cmap='Blues')

    # Add color bar
    plt.colorbar(cax)

    # Set axis labels
    ax.set_xticks(np.arange(len(all_chars)))
    ax.set_yticks(np.arange(len(all_chars)))

    # Label each axis with the character labels
    ax.set_xticklabels(all_chars, rotation=90)
    ax.set_yticklabels(all_chars)

    # Annotate each cell with the numeric value
    for i in range(len(all_chars)):
        for j in range(len(all_chars)):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='black')

    ax.set_xlabel('Predicted Characters')
    ax.set_ylabel('True Characters')
    ax.set_title('Character Confusion Matrix')

    plt.show()