import sys, cv2, os, time
import numpy as np, math
from argparse import ArgumentParser

from openvino.inference_engine import IENetwork, IEPlugin

from models.demo_onnx import *

input_size = 320
batch_size = 1
# These anchors are for yolov5s
anchors = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]



LABELS = ("person", "bicycle", "car", "motorbike", "aeroplane",
          "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird",
          "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack",
          "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat",
          "baseball glove", "skateboard", "surfboard","tennis racket", "bottle",
          "wine glass", "cup", "fork", "knife", "spoon",
          "bowl", "banana", "apple", "sandwich", "orange",
          "broccoli", "carrot", "hot dog", "pizza", "donut",
          "cake", "chair", "sofa", "pottedplant", "bed",
          "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven",
          "toaster", "sink", "refrigerator", "book", "clock",
          "vase", "scissors", "teddy bear", "hair drier", "toothbrush")
num_classes = 80

# LABELS = ('hand')
# num_classes = 1


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
                                                Sample will look for a suitable plugin for device specified (CPU by default)", default="CPU", type=str)
    return parser


def main_IE_infer():

    args = build_argparser().parse_args()
    model_xml = "weights/yolov5s.xml"
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    time.sleep(1)

    plugin = IEPlugin(device=args.device)
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    exec_net = plugin.load(network=net)

    image_path = "inference/images/zidane.jpg"

    start = time.time()
    image_src = Image.open(image_path)
    
    resized = letterbox_image(image_src, (input_size, input_size))
    img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0

    outputs = exec_net.infer(inputs={input_blob: img_in})
        
    # for _, value in outputs.items():
    #     print(value.shape)
    
    outputs = [value for _, value in outputs.items() ]
    batch_detections = []

    boxs = []
    a = torch.tensor(anchors).float().view(3, -1, 2)
    anchor_grid = a.clone().view(3, 1, -1, 1, 1, 2)
    
    if len(outputs) == 4:
        outputs = [outputs[1], outputs[2], outputs[3]]
    
    for index, out in enumerate(outputs):
        out = torch.from_numpy(out)
        batch = out.shape[1]
        feature_w = out.shape[2]
        feature_h = out.shape[3]

        # Feature map corresponds to the original image zoom factor
 
        stride_w = int(input_size / feature_w)
        stride_h = int(input_size / feature_h)

        grid_x, grid_y = np.meshgrid(np.arange(feature_w), np.arange(feature_h))

        # cx, cy, w, h
        pred_boxes = torch.FloatTensor(out[..., :4].shape)
        pred_boxes[..., 0] = (torch.sigmoid(out[..., 0]) * 2.0 - 0.5 + grid_x) * stride_w  # cx
        pred_boxes[..., 1] = (torch.sigmoid(out[..., 1]) * 2.0 - 0.5 + grid_y) * stride_h  # cy
        pred_boxes[..., 2:4] = (torch.sigmoid(out[..., 2:4]) * 2) ** 2 * anchor_grid[index]  # wh

        conf = torch.sigmoid(out[..., 4])
        pred_cls = torch.sigmoid(out[..., 5:])

        output = torch.cat((pred_boxes.view(batch_size, -1, 4),
                            conf.view(batch_size, -1, 1),
                            pred_cls.view(batch_size, -1, num_classes)),
                            -1)
        boxs.append(output)

    outputx = torch.cat(boxs, 1)

    # NMS
    batch_detections = w_non_max_suppression(outputx, num_classes, conf_thres=0.09, nms_thres=0.1)
    end = time.time()
    print(f"Processing time: {(end - start)}")

    if batch_detections[0] is not None:
        display(batch_detections[0], image_path, text_bg_alpha=0.6)


    del net
    del exec_net
    del plugin


if __name__ == '__main__':
    sys.exit(main_IE_infer() or 0)