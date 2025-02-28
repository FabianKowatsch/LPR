import cv2
import numpy as np

class Processing:
    def __init__(self, config):
        self.use_grayscale = config["grayscale"]
        self.denoising = config["denoising"]
        self.normalization = config["normalize"]
        self.enhance_contrast = config["contrast"]
        self.thresholding = config["thresholding"]
        self.threshold_value = config["threshold_value"]
        self.rotation = config["rotation"]
        self.max_angle = config["max_rotation_angle"]

    def __call__(self, image, box):

        if(self.denoising):
            image = self.denoise(image)
        if(self.normalization):
            image = self.normalize(image)
        if(self.use_grayscale):
            image = self.grayscale(image)
        #image = self.sharpen(image)
        if(self.enhance_contrast):
            image = self.contrast(image)
        if(self.rotation):
            image = self.rotate(image, box)
        if(self.thresholding):
            image = self.threshold(image)
        #image = cv2.Canny(image, 50, 150)
        if(self.use_grayscale):
            image =  cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image

    def grayscale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return gray
        #return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #statt COLOR_RGB2GRAY
    
    def denoise(self, image):
        #return cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)
        return cv2.bilateralFilter(image, 7, 1, 1)
        #return cv2.GaussianBlur(image, (7, 7), sigmaX=1)
    
    def sharpen(self, image):
        # Schärfungsfilter zum Hervorheben von Details
        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    
    def normalize(self, image):
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    def contrast(self, image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def threshold(self, image):
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_TOZERO, 11, 0.2)
        #_, image = cv2.threshold(image, self.threshold_value, 255, cv2.THRESH_BINARY)
        return image

    def rotate(self, image, box):

        height, width = image.shape[:2]

        x_min, x_max, y_min, y_max = get_min_max_values(box)
        x1, y1, x2, y2, x3, y3, x4, y4 = map(float, box.xyxyxyxy[0].flatten())
        corners = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        corners_px = np.array(corners, dtype=np.float32)
        corners = order_corners(corners_px)
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = corners

        # Compute widths and heights of the edges
        w1 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        w2 = np.sqrt((x3 - x4)**2 + (y3 - y4)**2)
        h1 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        h2 = np.sqrt((x4 - x1)**2 + (y4 - y1)**2)

        # The final width is the max of the two computed widths
        max_width = max(int(w1), int(w2))
        # The final height is the max of the two computed heights
        max_height = max(int(h1), int(h2))

        # Destination points in the target image (top-left, top-right, bottom-right, bottom-left)
        dst_pts = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(corners, dst_pts)

        # Warp the image
        warped = cv2.warpPerspective(image, M, (max_width, max_height))

        return warped

        # src_pts = np.array([
        #     [x1 - x_min, y1 - y_min], 
        #     [x2 - x_min, y2 - y_min],
        #     [x3 - x_min, y3 - y_min],
        #     [x4 - x_min, y4 - y_min] 
        # ], dtype=np.float32)

        # dst_pts = np.array([
        #     [width, height],
        #     [width, 0],
        #     [0, 0],
        #     [0, height]
        # ], dtype=np.float32)

        # Compute perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Apply warp perspective
        warped_image = cv2.warpPerspective(image, matrix, (int(width), int(height)))

        return warped_image
    

    def find_license_plate_corners(self, image):
 
        # 1. Vorverarbeitung: In Graustufen umwandeln und Rauschen reduzieren
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # 2. Kantenerkennung mit Canny
        edged = cv2.Canny(blurred, 50, 200)

        # 3. Konturfindung
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Sortiere Konturen nach Fläche (absteigend)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for contour in contours:
            # 4. Approximierung der Kontur
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Wenn die approximierte Kontur 4 Punkte hat, nehmen wir an, dass dies das Kennzeichen ist
            if len(approx) == 4:
                return approx.reshape(4, 2)
    
        return None

def get_min_max_values(box):
    x1, y1, x2, y2, x3, y3, x4, y4 = map(int, box.xyxyxyxy[0].flatten())

    ys = [y1, y2, y3, y4]
    xs = [x1, x2, x3, x4]
    return  np.min(xs),  np.max(xs),  np.min(ys),  np.max(ys)

def order_corners(corners):

    pts = corners.copy()
    # Compute centroid
    cx, cy = np.mean(pts, axis=0)
    print("pts: ", pts)
    # Compute angle of each point w.r.t. the centroid
    angles = np.arctan2(pts[:,1] - cy, pts[:,0] - cx)

    # Sort points by angle from -pi to pi
    sorted_idx = np.argsort(angles)
    pts_sorted = pts[sorted_idx]

    return pts_sorted


