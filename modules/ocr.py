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
        print(self.config)
        self.model_name = self.config["type"]
        self.model = None
        

        # Tesseract for windows
        if self.model_name == "tesseract":
            
            if os.name == "nt":
                 # Install if not available
                if not os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
                    tesseract_installer = r'tesseract-ocr-w64-setup-5.5.0.20241111.exe'
                    if os.path.exists(tesseract_installer):
                        subprocess.run([tesseract_installer, '/S'])
                    else:
                        raise FileNotFoundError("Tesseract installer not found at the specified location.")
            
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

            elif os.name == "posix":
                # Überprüfen, ob Tesseract installiert ist
                if not os.path.exists(r'/usr/local/bin/tesseract'):
                    raise FileNotFoundError("Tesseract is not installed at '/usr/local/bin/tesseract'. Please install it using Homebrew (brew install tesseract).")
            
                # Setze den Pfad zu Tesseract
                pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

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
        try:
            plate_text = pytesseract.image_to_string(image, config=self.tesseract_config)
            if not plate_text.strip():  # Kein Text erkannt
                raise ValueError("No text detected.")
            return plate_text.strip()
        except Exception as e:
            print(f"Tesseract OCR failed: {e}")  # Debugging-Ausgabe
            return "OCR failed: No text detected or invalid input."


    def ocr_easyocr(self, image):
        try:
            results = self.model.readtext(image)
            if not results or len(results) == 0:  # Kein Text erkannt
                raise ValueError("No text detected.")
            return " ".join([result[1] for result in results])
        except Exception as e:
            print(f"EasyOCR failed: {e}")  # Debugging-Ausgabe
            return "OCR failed: No text detected or invalid input."
        

    def ocr_parseq(self, image):
        try:
            img = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            img /= 255.0
            
            transform = self.get_parseq_transform(self.img_size)
            img = transform(img).unsqueeze(0)

            logits = self.model(img)
            pred = logits.softmax(-1)
            label, confidence = self.model.tokenizer.decode(pred)

            if not label or len(label[0].strip()) == 0:  # Kein Text erkannt
                raise ValueError("No text detected.")
            print("TEXT", repr(label[0]))
            print("confidence", confidence)
            filtered_text = ""
            for i, (char, conf) in enumerate(zip(label[0], confidence[0])):
                if conf < 0.5:
                    continue
                if i in [1, 2, 3] and conf < 0.6:
                    continue
                filtered_text += char
            print("FILTERED", filtered_text)
            return filtered_text
        except Exception as e:
            print(f"Parseq OCR failed: {e}")  # Debugging-Ausgabe
            return "OCR failed: No text detected or invalid input."
    
    def get_parseq_transform(self, img_size: tuple[int] = (32, 128)):
        transforms = []
        transforms.extend([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.Normalize(0.5, 0.5),
        ])
        return T.Compose(transforms)
