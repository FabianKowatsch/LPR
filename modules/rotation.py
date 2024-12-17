import cv2

class Rotation:
    def __init__(self):
        self.max_angle = 15

    def __call__(self, image):

        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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

        h,w,_ = image.shape

        rotation = cv2.getRotationMatrix2D(rect[0], angle, 1.0)

        aligned_image = cv2.warpAffine(image, rotation, (w, h))

        return aligned_image