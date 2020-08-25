
# Remove cached labels
rm  ./datasets/CEOT_hand_yolo/labels/*.cache

python train.py \
        --img 640 \
        --cfg ./models/yolov5l_ceot_hand.yaml \
        --batch 1 \
        --epochs 50 \
        --data ./data/ceot_hand.yaml \
        --single-cls \
        --weights ./weights/yolov5l.pt\
        --device 5

