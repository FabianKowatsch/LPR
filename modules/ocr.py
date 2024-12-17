import torch
import pytesseract
import easyocr
from torchvision import transforms as T
import matplotlib.pyplot as plt
import os
import subprocess

class OCR_Module:
    def __init__(self, config):
        self.config = config
        self.model_name = self.config["recognizer"]
        self.model = None
        
        # Tesseract
        if self.model_name == "tesseract":

            # Install if not available
            if not os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
                tesseract_installer = r'tesseract-ocr-w64-setup-5.5.0.20241111.exe'
                if os.path.exists(tesseract_installer):
                    subprocess.run([tesseract_installer, '/S'])
                else:
                    raise FileNotFoundError("Tesseract installer not found at the specified location.")
            
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

            # Check if installation was successful
            try:
                tesseract_version = pytesseract.get_tesseract_version()
                print(f"Tesseract version: {tesseract_version}")
            except pytesseract.TesseractNotFoundError:
                print("Tesseract is not installed or not added to the PATH.")
            self.forward = self.ocr_tesseract
            tesseract_oem = config["tesseract_engine"]
            tesseract_psm = config["tesseract_segmentation"]
            self.tesseract_config = f'--oem {tesseract_oem} --psm {tesseract_psm}'

        # EasyOCR
        elif self.model_name == "easyocr":
            self.model = easyocr.Reader([self.config["language"]])
            self.forward = self.ocr_easyocr
        
        # Parseq
        elif self.model_name == "parseq":
            self.model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
            self.forward = self.ocr_parseq
            self.img_size = self.config["parseq_img_size"]

    def __call__(self, image):
        return self.forward(image)
    
    def ocr_tesseract(self, image):
        plate_text = pytesseract.image_to_string(image, config=self.tesseract_config)
    
        return plate_text.strip()

    
    def ocr_easyocr(self, image):
        results = self.model.readtext(image)
        return " ".join([result[1] for result in results])
    
    def ocr_parseq(self, image):
        img = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        img /= 255.0
        
        transform = self.get_parseq_transform(self.img_size)
        img = transform(img).unsqueeze(0)

        logits = self.model(img)
        pred = logits.softmax(-1)
        label, confidence = self.model.tokenizer.decode(pred)

        return label[0]
    
    def get_parseq_transform(self, img_size: tuple[int] = (32, 128)):
        transforms = []
        transforms.extend([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.Normalize(0.5, 0.5),
        ])
        return T.Compose(transforms)
