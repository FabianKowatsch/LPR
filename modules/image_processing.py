import cv2

class Processing:
    def __init__(self, config):
        self.use_grayscale = config["grayscale"]
        self.denoising = config["denoising"]
        self.normalize = config["normalize"]
        self.enhance_contrast = config["contrast"]
        self.thresholding = config["thresholding"]
        self.threshold_value = config["threshold_value"]
        self.rotation = config["rotation"]
        self.max_angle = config["max_rotation_angle"]

    def __call__(self, image):

        if(self.use_grayscale):
            image = self.grayscale(image)
        if(self.denoising):
            image = self.denoise(image)
        if(self.enhance_contrast):
            image = self.contrast(image)
        if(self.thresholding):
            image = self.threshold(image)
        if(self.rotation):
            image = self.rotate(image)
        if(self.use_grayscale):
            image =  cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    def grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    def denoise(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    def normalize(self, image):
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    def contrast(self, image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def threshold(self, image):
        _, image = cv2.threshold(image, self.threshold_value, 255, cv2.THRESH_BINARY)
        return image

    def rotate(self, image):
        grayscale = image if self.use_grayscale else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Edge detection
        edges = cv2.Canny(grayscale, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image
        
        valid_contours = []

        for contour in contours:
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)
        
            # Aspect ratio filtering: the contour should be rectangular
            aspect_ratio = float(w) / h
        
            # Only consider contours with an appropriate aspect ratio
            if 2 < aspect_ratio < 6:  # Adjust these values according to your needs
                valid_contours.append(contour)

        if not valid_contours:
            return image
        
        # Select the largest contour
        c = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(c)
        angle = rect[2]

        # If the angle is too big, the contour was probably wrong
        if abs(angle) > self.max_angle:
            return image

        #h,w,_ = image.shape

        rotation = cv2.getRotationMatrix2D(rect[0], angle, 1.0)

        aligned_image = cv2.warpAffine(image, rotation, (image.shape[1], image.shape[0]))

        return aligned_image