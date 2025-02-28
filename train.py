from modules.detection import LPD_Module

def train():
    detector = LPD_Module("checkpoints/pretrained/yolo11n.pt")
    #results = detector.model.train(data="config/data.yaml", epochs=400, imgsz=640, device=0, val=True, single_cls=True, batch=32)
    results = detector.model.train(cfg="config/train.yaml")
    #print(results)

def train_obb():
    detector = LPD_Module("yolo11n-obb.pt", verbose=False)
    #results = detector.model.train(data="config/data.yaml", epochs=400, imgsz=640, device=0, val=True, single_cls=True, batch=32)
    results = detector.model.train(cfg="config/train_obb.yaml")
if __name__ == "__main__":
    train_obb()