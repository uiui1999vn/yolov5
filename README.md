

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

## Data preparation

Please download and put the [CEOT](https://drive.google.com/file/d/1UOG_XrZ8ZlFBvQOHFk7AHzx4m0tz4EYU/view?usp=sharing) hand dataset and put into the folder `yolov5/datasets`.

```
yolov5/datasets/CEOT_hand_yolo/
├── images
│   ├── test
│   ├── train
│   └── val
└── labels
    ├── test
    ├── train
    └── val
```

## Download pretrained models on COCO

For compatibility with the provided training code, please download these pretrained [Weights](https://drive.google.com/file/d/1NoFDMlFNTzBeUsis09vNJH7Wv0zbUv7L/view?usp=sharing), and put into the `yolov5/weights`. The folder structure looks like this:

```
yolov5/weights/
├── yolov5l.pt
├── yolov5m.pt
├── yolov5s.pt
└── yolov5x.pt
```


## Test

After training, evaluate the performance of a trainned model on the CEOT hand test set.
```bash
$ ./run_test.sh
```

## Training

Run command below.
```bash
$ ./run_train.sh                                   
```
