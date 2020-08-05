

Train and evaluate YOLOv5 on the fisheye person-basket detection dataset.

## Comparison

`Speed/FPS` was measured on an NVIDIA GPU RTX2018 Ti @ `batch_size=1`

`image_size=512`

| Model | AP<sup>test</sup> | Precision| Recall | Speed<sub>GPU</sub> | FPS<sub>GPU</sub> || params | FLOPS |
|---------- |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| YOLOv5m | 87.4     | 93.3   | 95.2    | 19.4ms     | 51     || 21.8M  | 39.4B


## Requirements

Python 3.8 or later with all `requirements.txt` dependencies installed. To install run:
```bash
$ conda create -n yolov5 python=3.8
$ conda activate yolov5
$ conda install pip
$ pip install -r requirements.txt
```

## Data preparation

Please download and put the dataset and put into the folder `yolov5/datasets`.

```
yolov5/datasets/person_basket/
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

Evaluate the performance of a trainned model on the test set.
```bash
$ ./run_test.sh
```

## Training

Run command below.
```bash
$ ./run_train.sh                                   
```
