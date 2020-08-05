

Train and evaluate YOLOv5 on CEOT Hand dataset.

## Comparison

`Speed/FPS` was measured on an NVIDIA GPU RTX2018 Ti @ `batch_size=1`

`image_size=640`

| Model | AP<sup>test</sup> | Precision| Recall | Speed<sub>GPU</sub> | FPS<sub>GPU</sub> || params | FLOPS |
|---------- |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| YOLOv5l | 46.5     | 62.1   | 86.0    | 20.9ms     | 48     || 47.8M  | 88.1B


## Requirements

Python 3.8 or later with all `requirements.txt` dependencies installed. To install run:
```bash
$ conda create -n yolov5 python=3.8
$ conda activate yolov5
$ conda install pip
$ pip install -r requirements.txt
```

## Inference

Inference can be run on most common media formats: images, videos, please check all the options.
```bash
$ ./run_detect.sh
```

## Test

Evaluate the performance of a trainned model on the Hand test set.
```bash
$ ./run_test.sh
```

## Training

Run command below.
```bash
$ ./run_train.sh                                   
```
