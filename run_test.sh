: '
batch_size = 1
'

# Remove cached labels
rm /home/thanhnv/Datasets/sku110_yolo/labels/*.npy

python test.py \
        --weights ./weights/best.pt \
        --data ./data/sku110.yaml \
        --batch 1 \
        --img 736 \
        --task 'test' \
        --single-cls \
        --device 3 \
        --verbose
