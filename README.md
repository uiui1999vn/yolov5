Convert yolov5 models (verson 2) in `*.pt` format to `OpenVINO`, there are 2 steps:

## 1. Convert to ONNX

### Requirements

Just use `yolov5 verson 2` requirements, and the following packages:

```bash
$ pip install onnx==1.7.0
$ pip install onnxruntime==1.4.0
```

### Conversion

```bash
$ ./run_export_onnx.sh
```

### Verification

```bash
$ cd <YOLOv5_INSTALL_DIR>
$ python models/demo_onnx.py
```

## 2. Convert from ONNX to OpenVINO

### OpenVINO installation

Install `OpenVINO` 2020.4 in a `USER` account.

### Configuration

Please use `python 3.6`

```bash
$ cd <INSTALL_DIR>/deployment_tools/model_optimizer/
$ virtualenv -p python3 openvino --system-site-packages
$ source openvino/bin/activate
$ pip3 install -r requirements_onnx.txt
```

### Conversion

```bash
$ cd <YOLOv5_INSTALL_DIR>
$ source openvino/bin/activate
$ source ~/intel/openvino/bin/setupvars.sh
$ ./run_convert.sh
```

### Verification

```bash
$ cd <YOLOv5_INSTALL_DIR>
$ python models/demo_openvino.py
```




