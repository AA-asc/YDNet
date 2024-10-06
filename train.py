import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/yolo-detr/yolov8-detr.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=8,
                workers=8,
                device='0',
                # resume='', # last.pt path
                project='runs/train',
                name='exp',
                )