:'
batch_size = 1
'
# Remove cached labels
rm /Users/thanhnguyen/Documents/Datasets/COCO2017_Person/labels/*.cache

python test.py \
        --weights weights/org/best_xs.pt \
        --data ./data/sku110k.yaml \
        --batch 1 \
        --img 320 \
        --task 'test' \
        --single-cls \
        --device 'cpu' \
        --verbose
