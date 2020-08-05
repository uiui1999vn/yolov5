: '
batch_size = 1
'

# Remove cached labels
rm /home/thanhnv/Datasets/CEOT_hand_yolo/labels/*.cache

python test.py \
        --weights ./runs/exp2/weights/best.pt \
        --data ./data/ceot_hand.yaml \
        --batch 1 \
        --img 640 \
        --task 'test' \
        --single-cls \
        --device 5 \
        --verbose
