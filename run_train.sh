
# Remove cached labels
rm  /home/thanhnv/Documents/Datasets/COCO2017/labels/*.cache

python train.py \
        --img 320 \
        --cfg ./models/yolov5s.yaml \
        --batch 180 \
        --epochs 200 \
        --data ./data/coco_person.yaml \
        --single-cls \
        --weights ' ' \
        --device 0,1,2 \
        --multi-scale \
        --sync-bn


