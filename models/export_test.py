import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

import onnx
import onnxruntime

class Test(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.interpolate(x, size=(400, 600), mode='bilinear', align_corners=False) #no warning, all clear

model = Test()
x = torch.rand((1, 3, 200, 300))
torch.onnx._export(model, x, "weights/test.onnx", verbose=True)

model.eval()
with torch.no_grad():
    torch_out = model(x)

ort_session = onnxruntime.InferenceSession("weights/test.onnx")
ort_input = {ort_session.get_inputs()[0].name: x.cpu().numpy()}
ort_out = ort_session.run(None, ort_input)[0]
