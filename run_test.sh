: '
batch_size = 1
'

# Remove cached labels
rm /mnt/ssd2/Datasets/Fish_eye_dataset/PJ9/ROI_PJ9_person360/labels.cache

python test.py \
        --weights /home/anhvn/yolov5_new/best.pt \
        --data ./data/person_basket_3cls.yaml \
        --batch 1 \
        --img 512 \
        --task 'test' \
        --single-cls \
        --device 2 \
        --verbose
