import cv2
import numpy as np
from utils import get_min_max_values, sort_rectangle_points, create_color_mask

class Processing:
    def __init__(self, config):
        self.use_grayscale = config["grayscale"]
        self.denoising = config["denoising"]
        self.normalization = config["normalize"]
        self.color_mask = config["color_mask"]
        self.thresholding = config["thresholding"]
        self.threshold_value = config["threshold_value"]
        self.rotation = config["rotation"]

    def __call__(self, image, box, scale_factor):

        #image = self.white_balance(image)
        if(self.denoising):
            image = self.denoise(image)

        if(self.color_mask):
            image = self.mask_color(image)

        if(self.normalization):
            image = self.normalize(image)

        if(self.use_grayscale):
            image = self.grayscale(image)

        if(self.rotation):
            image = self.rotate(image, box, scale_factor)

        if(self.thresholding):
            image = self.threshold(image)

        if(self.use_grayscale):
            image =  cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image

    def grayscale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # cv2.COLOR_RGB2GRAY
        gray = cv2.equalizeHist(gray)
        return gray
    
    def denoise(self, image):
        #return cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)
        #return cv2.GaussianBlur(image, (7, 7), sigmaX=1)
        return cv2.bilateralFilter(image, 7, 1, 1)
    
    def sharpen(self, image):
        # Schärfungsfilter zum Hervorheben von Details
        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    
    def mask_color(self, image):
        # Get the individual color masks
        too_red_mask, too_green_mask, too_blue_mask = create_color_mask(image, 100, 1.5)

        height, width, _ = image.shape
        border_width = width // 8
        left_right_mask = np.zeros((height, width), dtype=bool)
        left_right_mask[:, :border_width] = True
        left_right_mask[:, -border_width:] = True

       # Combine all masks (too red, too green, too blue, and left-right mask) in one line
        final_mask = (too_red_mask | too_green_mask | too_blue_mask) & left_right_mask

        num_pixels = np.sum(final_mask)
        if num_pixels > 0:
            # Generate (num_pixels, 3) random noise
            random_noise = np.random.randint(180, 256, size=(num_pixels, 3), dtype=np.uint8)

            # Apply noise directly using the mask (fancy indexing trick)
            image[final_mask] = random_noise
        return image
    
    def normalize(self, image):
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    def threshold(self, image):
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_TOZERO, 11, 0.2)
        #_, image = cv2.threshold(image, self.threshold_value, 255, cv2.THRESH_BINARY)
        return image

    def rotate(self, image, box, scale_factor):
        height, width = image.shape[:2]
        x_min, x_max, y_min, y_max = get_min_max_values(box)
        x1, y1, x2, y2, x3, y3, x4, y4 = map(int, box.xyxyxyxy[0].flatten())

        pts = np.array([
        [x1-x_min, y1-y_min],
        [x2-x_min, y2-y_min],
        [x3-x_min, y3-y_min],
        [x4-x_min, y4-y_min] 
        ], dtype=np.float32)

        src_pts = sort_rectangle_points(pts)
        #src_pts = src_pts.astype(np.int32)     
        src_pts = scale_factor* src_pts
        
        dst_pts = np.array([
            [0,0], 
            [x_max -x_min, 0],
            [x_max -x_min, y_max - y_min],
            [0, y_max - y_min]
            ])
        dst_pts = scale_factor* dst_pts

        matrix = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
        image = cv2.warpPerspective(image, matrix, (width, height))

        return image
    
    def white_balance(self, image):
        # Convert image to float32 for precise calculations
        image_float = image.astype(np.float32)

        # Find the whitest pixel (maximum values across channels)
        max_rgb = np.max(image_float, axis=(0, 1))  # Max value for each channel

        # Calculate scaling factors for each channel (to make the whitest point white)
        scaling_factors = 255.0 / max_rgb

        # Apply the scaling factors to the image
        balanced_image = image_float * scaling_factors

        # Clip values to ensure they stay within the valid range [0, 255]
        balanced_image = np.clip(balanced_image, 0, 255)

        # Convert the image back to uint8
        balanced_image = balanced_image.astype(np.uint8)

        return balanced_image



