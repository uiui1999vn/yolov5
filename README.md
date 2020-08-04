

Train and evaluate YOLOv5 on SKU110K dataset.


## Comparison

`Speed/FPS` was measured on an NVIDIA GPU RTX2018 Ti @ `batch_size = 1`

`image_size = 640`

| Model | AP<sup>test</sup> | Precision| Recall | Speed<sub>GPU</sub> | FPS<sub>GPU</sub> || params | FLOPS |
|---------- |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| YOLOv5m | 55.2     | 57.2     | 92.3    | 17ms     | 58     || 21.8M  | 39.4B
| YOLOv5l | 55.4     | 56.3     | 92.3    | 19ms     | 52     || 47.8M  | 88.1B


`image_size = 736`

| Model | AP<sup>test</sup> | Precision| Recall | Speed<sub>GPU</sub> | FPS<sub>GPU</sub> || params | FLOPS |
|---------- |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| YOLOv5m | 56.2     | 58.9     | 93.2    | 31ms     | 32     || 21.8M  | 39.4B
| YOLOv5l | 56.6     | 59.9     | 93.2    | 28ms     | 35     || 47.8M  | 88.1B


`image_size = 1280`

| Model | AP<sup>test</sup> | Precision| Recall | Speed<sub>GPU</sub> | FPS<sub>GPU</sub> || params | FLOPS |
|---------- |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| YOLOv5m | 58.2     | 64.4     | 92.4    | 45ms     | 22     || 21.8M  | 39.4B
| YOLOv5l | 58.1     | 64.8     | 94.3    | 67ms     | 15     || 47.8M  | 88.1B


`image_size = 1408`

| Model | AP<sup>test</sup> | Precision| Recall | Speed<sub>GPU</sub> | FPS<sub>GPU</sub> || params | FLOPS |
|---------- |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| YOLOv5m | 58.4     | 66.3     | 94.6    | 30ms     | 33     || 21.8M  | 39.4B
| YOLOv5l |      |      |     |      |      || 47.8M  | 88.1B


## Requirements

`Python 3.7` or later with all `requirements.txt` dependencies installed. To install run:
```bash
$ conda create -n yolov5 python=3.7
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

Evaluate the performance of a trainned model on the SKU110K test set.
```bash
$ ./run_test.sh
```

## Training

Run command below.
```bash
$ ./run_train.sh                                   
```
