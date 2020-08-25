
# Remove cached labels
rm  /home/thanhnv/Documents/Datasets/COCO2017/labels/*.cache

python train.py \
        --img 512 \
        --cfg ./models/yolov5s.yaml \
        --batch 32 \
        --epochs 100 \
        --data ./data/coco_person.yaml \
        --single-cls \
        --weights weights/yolov5s.pt \
        --device 0,1 \
        --multi-scale \
        --sync-bn


