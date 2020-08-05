

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

Please download and put the dataset into the folder `yolov5/datasets`.

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
