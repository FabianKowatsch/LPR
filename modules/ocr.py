import torch
import pytesseract
import easyocr
from torchvision import transforms as T
import matplotlib.pyplot as plt

class OCR_Module:
    def __init__(self, config):
        self.config = config
        self.model_name = self.config["recognizer"]
        self.recognize = None
        self.model = None
        if self.model_name == "tesseract":
            try:
                tesseract_version = pytesseract.get_tesseract_version()
                print(f"Tesseract version: {tesseract_version}")
            except pytesseract.TesseractNotFoundError:
                print("Tesseract is not installed or not added to the PATH.")
            self.recognize = self.ocr_tesseract
        elif self.model_name == "easyocr":
            self.model = easyocr.Reader([self.config["language"]])
            self.recognize = self.ocr_easyocr
        elif self.model_name == "parseq":
            self.model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
            self.recognize= self.ocr_parseq
            self.img_size = self.config["parseq_img_size"]

    
    def ocr_tesseract(self, image):
        custom_config = r'--oem 3 --psm 8'
        plate_text = pytesseract.image_to_string(image, config=custom_config)
        print(f"Recognized License Plate Text: {plate_text}")
    
        return plate_text.strip(), None, None, None

    
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
