: '
batch_size = 1
'

# Remove cached labels
rm /home/thanhnv/Datasets/sku110_yolo/labels/*.cache
EXP=exp1

python test.py \
        --weights runs/${EXP}/weights/best.pt \
        --data ./data/sku110.yaml \
        --batch 1 \
        --img 1408 \
        --task 'test' \
        --single-cls \
        --device 0 \
        --verbose
