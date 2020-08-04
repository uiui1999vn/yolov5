
# Remove cached labels
rm /home/thanhnv/Datasets/sku110_yolo/labels/*.cache

python train.py \
        --img 1408 \
        --cfg ./models/yolov5m.yaml \
        --batch 4 \
        --epochs 60 \
        --data ./data/sku110.yaml \
        --single-cls \
        --weights ./weights/yolov5m.pt\
        --device 5

