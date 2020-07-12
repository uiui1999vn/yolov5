"""
Load model by using state dict
"""

import torch
import cv2
import numpy as np
import math
import glob

from models.yolo import Model
from utils.datasets import *
from utils.utils import *
from utils.video_utils import get_vid_properties, VideoWriter

# set True to speed up constant image size inference
torch.backends.cudnn.benchmark = True  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def load_checkpoint(checkpoint_path, model, optimizer=None, device=None, train=False):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.fuse()
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    if device is not None:
        model.to(device)
    if not train:
        model.eval()
    
    return model

def preprocess_img(img_BGR, imgsz=640, device=device):
    # Padded resize
    img = letterbox(img_BGR, new_shape=imgsz)[0]
    # Convert
    img_RGB = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img_RGB)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


imgsz = 736
model_cfg = 'yolov5m'
ckpt = f'ckpt_65_{imgsz}'
cfg = f'/home/thanhnv/Detection/yolov5/models/{model_cfg}.yaml'
weights = f'/home/thanhnv/Detection/yolov5/weights/sku110/{model_cfg}/{ckpt}.pth'


imgsz = check_img_size(imgsz)
conf_thres = 0.3
iou_thres = 0.5
label = ['object']
color = (255, 0, 0)

# Create model
model = Model(cfg)
model = load_checkpoint(weights, model, device=device)

@torch.no_grad()
def predict():
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.float()) if device.type != 'cpu' else None  # run once

    img_path = '/mnt/ssd2/Datasets/RPC/val2019'
    out = '/home/thanhnv/Detection/yolov5/inference/output/RPC'

    img_files = glob.glob(img_path + '/*.jpg')
    img_files.sort()

    for img_file in img_files[:10]:
        img_BGR =  cv2.imread(img_file)
        im0 = img_BGR.copy()
        img = preprocess_img(img_BGR, imgsz=imgsz)

        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)

        for i, det in enumerate(pred):
            # det = [x0, y0, x2, y2, conf, cls]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cat in det:
                    plot_one_box(xyxy, im0, label=label[0], color=color, line_thickness=3)
                

        path = str(Path(img_file)) 
        save_path = str(Path(out) / Path(path).name)
        cv2.imwrite(save_path, im0)

if __name__ == '__main__':
    predict()




