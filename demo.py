
import torch
import cv2
import numpy as np
import math

from utils.datasets import *
from utils.utils import *

# set True to speed up constant image size inference
torch.backends.cudnn.benchmark = True  
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

weights = '/home/thanhnv/Detection/yolov5/runs/exp0/weights/best.pt'
imgsz = 736
imgsz = check_img_size(imgsz)
conf_thres = 0.4
iou_thres = 0.5
label = ['obj']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(label))]

def load_model(weights, device):
    ckpt = torch.load(weights, map_location='cpu')
    epoch = ckpt['epoch']
    print(f'epoch: {epoch}')
    model = ckpt['model']
    model.to(device).eval()
    return model

@torch.no_grad()
def predict(model):
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.float()) if device.type != 'cpu' else None  # run once

    img_file = '/home/thanhnv/Datasets/sku110_yolo/images/test/test_711.jpg'
    out = '/home/thanhnv/Detection/yolov5/inference/output'
    path = str(Path(img_file)) 
    save_path = str(Path(out) / Path(path).name)

    img_BGR =  cv2.imread(img_file)
    im0 = img_BGR.copy()
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

    print(model)
    
    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)

    for i, det in enumerate(pred):
        # det = [x0, y0, x2, y2, conf, cls]
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cat in det:
                plot_one_box(xyxy, im0, label=label[0], color=colors[0], line_thickness=3)
            
        cv2.imwrite(save_path, im0)
    
if __name__ == '__main__':
    predict(load_model(weights, device))