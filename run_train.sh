
# Remove cached labels
rm /home/thanhnv/Datasets/sku110_yolo/labels/*.cache

# python train.py \
#         --img 1600 \
#         --cfg ./models/yolov5m.yaml \
#         --batch 4 \
#         --epochs 40 \
#         --data ./data/sku110.yaml \
#         --single-cls \
#         --weights ./weights/yolov5m.pt\
#         --device 1,5

python train.py \
        --img 736 \
        --cfg ./models/yolov5l.yaml \
        --batch 4 \
        --epochs 80 \
        --data ./data/sku110.yaml \
        --single-cls \
        --weights ./weights/yolov5l.pt\
        --device 3

