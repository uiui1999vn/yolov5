
# Remove cached labels
rm  /home/thanh_nguyen/Datasets/COCO2017_Person/labels/*.cache

python train.py \
        --img 320 \
        --cfg ./models/yolov5s.yaml \
        --batch 128 \
        --epochs 1000 \
        --data ./data/coco_person.yaml \
        --single-cls \
        --weights ' ' \
        --device 0 \
        --multi-scale \


