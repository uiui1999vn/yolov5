# Convert yolov5 models (verson 2) in `*.pt` format to `OpenVINO`

## 1. Convert to ONNX

### 1.1. Requirements

Just use `yolov5 verson 2` requirements, and the following packages:

```bash
$ pip install onnx==1.7.0
$ pip install onnxruntime==1.4.0
```

### 1.2. Conversion

```bash
$ ./run_export_onnx.sh
```

### 1.3. Verification

```bash
$ cd <YOLOv5_INSTALL_DIR>
$ python models/demo_onnx.py
```

## 2. Convert from ONNX to OpenVINO

### 2.1. OpenVINO installation

Install `OpenVINO` 2020.4 in a `USER` account.

### 2.2. Configuration

Please use `python 3.6`

```bash
$ cd <INSTALL_DIR>/deployment_tools/model_optimizer/
$ virtualenv -p python3 openvino --system-site-packages
$ source openvino/bin/activate
$ pip3 install -r requirements_onnx.txt
```

### 2.3. Conversion

```bash
$ cd <YOLOv5_INSTALL_DIR>
$ source openvino/bin/activate
$ source ~/intel/openvino/bin/setupvars.sh
$ ./run_convert.sh
```

### 2.4. Verification

```bash
$ cd <YOLOv5_INSTALL_DIR>
$ python models/demo_openvino.py
```




