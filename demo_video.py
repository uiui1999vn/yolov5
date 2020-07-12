"""
Load model by using state dict
"""

import torch
import cv2
import numpy as np
import math

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

cfg = '/home/thanhnv/Detection/yolov5/models/yolov5l.yaml'
weights = '/home/thanhnv/Detection/yolov5/weights/sku110/yolov5l/ckpt_39_1280.pth'
imgsz = 1280
imgsz = check_img_size(imgsz)
conf_thres = 0.5
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

    vid_fname = '/home/thanhnv/Detection/yolov5/inference/CAM_WEST.mp4'
    save_dir = '/home/thanhnv/Detection/yolov5/inference/output'

    vid = cv2.VideoCapture(vid_fname)
    width, height, fps, num_frames = get_vid_properties(vid)
    basename = vid_fname.split('/')[-1]
    vid_writer = VideoWriter(width, height, fps, save_dir, basename)

    mean_time = 0
    while vid.isOpened():
        current_time = cv2.getTickCount()
        grabbed, img_BGR = vid.read()
        if not grabbed: break

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
                
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
        cv2.putText(im0, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                    (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
        
        vid_writer.write(im0)

    vid.release()
    vid_writer.release()


if __name__ == '__main__':
    predict()




