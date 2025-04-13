import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil

# Path to your image folder
IMAGE_FOLDER = 'data/valid/images'

# Output folder (can be same as image folder or different)
OUTPUT_FOLDER = 'data/obb/valid/labels'
LABEL_FOLDER = 'data/valid/labels'  

LABELED_FOLDER = 'data/obb/valid/images'

# Global variable to store clicked points
clicked_points = []

def load_bbox_labels(image_name):
    """Load all bboxes for an image (supports multiple objects).
    If no valid bboxes are found, return an empty list (no zoom).
    """
    label_file = os.path.join(LABEL_FOLDER, f"{os.path.splitext(image_name)[0]}.txt")

    if not os.path.exists(label_file):
        return []  # No label file found, treat as no objects (full image shown)

    bboxes = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()

            # Must have exactly 5 parts: class_index, center_x, center_y, width, height
            if len(parts) != 5:
                print(f"Skipping faulty line in {label_file}: {line}")
                continue

            try:
                _, center_x, center_y, width, height = map(float, parts)

                # Check if coordinates and size make sense
                if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and width > 0 and height > 0):
                    print(f"Skipping out-of-bounds bbox in {label_file}: {line}")
                    continue

                bboxes.append((center_x, center_y, width, height))

            except ValueError:
                print(f"Skipping invalid line (parse error) in {label_file}: {line}")
                continue

    return bboxes

def zoom_image(img, bbox, zoom_factor=3.0):
    """Zoom into bbox area with a bit of padding."""
    center_x, center_y, width, height = bbox

    img_h, img_w = img.shape[:2]
    center_x *= img_w
    center_y *= img_h
    width *= img_w
    height *= img_h

    # Apply zoom factor to widen the crop area (weaker zoom)
    zoom_w = width * zoom_factor
    zoom_h = height * zoom_factor

    # Calculate crop region (clamped to image boundaries)
    x1 = max(int(center_x - zoom_w / 2), 0)
    y1 = max(int(center_y - zoom_h / 2), 0)
    x2 = min(int(center_x + zoom_w / 2), img_w)
    y2 = min(int(center_y + zoom_h / 2), img_h)

    cropped_img = img[y1:y2, x1:x2]
    return cropped_img, (x1, y1, x2, y2)

def onclick(event):
    """Handle mouse clicks for getting corner points."""
    if event.xdata is not None and event.ydata is not None:
        clicked_points.append((event.xdata, event.ydata))
        plt.scatter(event.xdata, event.ydata, color='red')
        plt.draw()

        if len(clicked_points) == 4:
            plt.close()

def annotate_object(img, bbox):
    """Zoom to bbox and collect 4 corner points."""
    global clicked_points
    clicked_points = []

    cropped_img, (x1, y1, x2, y2) = zoom_image(img, bbox)

    fig, ax = plt.subplots()
    ax.imshow(cropped_img)
    ax.set_title(f"Click 4 corners (Object {current_object_index + 1})")

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(clicked_points) != 4:
        print("ERROR: Expected 4 points, got", len(clicked_points))
        return None

    # Convert clicked points in cropped view back to full image coordinates
    original_points = [(x + x1, y + y1) for x, y in clicked_points]

    # Normalize points to [0,1] relative to full image size
    img_h, img_w = img.shape[:2]
    normalized_points = [(x / img_w, y / img_h) for x, y in original_points]

    return normalized_points

def save_points(image_path, all_points):
    """Save all objects' points into a single file."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_file = os.path.join(OUTPUT_FOLDER, f"{base_name}.txt")

    with open(txt_file, 'w') as f:
        for points in all_points:
            line = "0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in points) + "\n"
            f.write(line)

    print(f"Saved: {txt_file}")

def process_image(image_path):
    global current_object_index

    img = mpimg.imread(image_path)
    image_name = os.path.basename(image_path)

    bboxes = load_bbox_labels(image_name)

    if not bboxes:
        print(f"No valid bboxes found for {image_name}, showing full image.")
        bboxes = [(0.5, 0.5, 1.0, 1.0)]  

    all_points = []

    for current_object_index, bbox in enumerate(bboxes):
        points = annotate_object(img, bbox)
        if points:
            all_points.append(points)
        else:
            print(f"Skipping object {current_object_index + 1} in {image_name}")

    if all_points:
        save_points(image_path, all_points)

def main():
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    images.sort()

    for image in images:
        image_path = os.path.join(IMAGE_FOLDER, image)
        process_image(image_path)

def cleanup():
    labeled_images = set()
    for file in os.listdir(OUTPUT_FOLDER):
        if file.endswith('.txt'):
            base_name = os.path.splitext(file)[0]
            labeled_images.add(base_name)

    # Loop through images and copy the labeled ones
    for image_file in os.listdir(IMAGE_FOLDER):
        if image_file.lower().endswith(('jpg', 'jpeg', 'png')):
            base_name = os.path.splitext(image_file)[0]
            if base_name in labeled_images:
                src_path = os.path.join(IMAGE_FOLDER, image_file)
                dest_path = os.path.join(LABELED_FOLDER, image_file)
                shutil.copy2(src_path, dest_path)
            print(f"Copied: {image_file}")

    print(f"Finished copying {len(labeled_images)} labeled images to {LABELED_FOLDER}.")

if __name__ == "__main__":
    #current_object_index = 0
    main()
    cleanup()