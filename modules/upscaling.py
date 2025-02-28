from torchvision import transforms as T
import cv2

class Upscaler:
    def __init__(self, config):
        self.mode = config["type"]
        self.scale_factor = config["scale_factor"]
        if(self.mode == "bicubic"):
            self.forward = self.bicubic
        elif(self.mode == "bilinear"):
            self.forward = self.bilinear
        elif(self.mode == "LANCZOS4"):
            self.forward = self.lanczos4
        #elif(self.mode == "GAN"):
            #model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            #model_path = "./checkpoints/realesrgan/RealESRGAN_x4plus.pth"
            #self.model = RealESRGANer(device='cuda', model=model, model_path=model_path, scale=self.scale_factor)
            #self.model.load_weights()
            #self.forward = self.esrgan
        else:
            raise ValueError(f"Unsupported upscaling mode: {self.mode}")
        
    def __call__(self, image):
        return self.forward(image)
    
    def bicubic(self, image):
        h, w, _ = image.shape
        return cv2.resize(image, (w * self.scale_factor, h * self.scale_factor), interpolation=cv2.INTER_CUBIC)
    
    def bilinear(self, image):
        h, w, _ = image.shape
        return cv2.resize(image, (w * self.scale_factor, h * self.scale_factor), interpolation=cv2.INTER_LINEAR, )
    
    def lanczos4(self, image):
        h, w, _ = image.shape
        return cv2.resize(image, (w * self.scale_factor, h * self.scale_factor), interpolation=cv2.INTER_LANCZOS4)
    
    #def esrgan(self, image):
        #return self.model.predict(image)
