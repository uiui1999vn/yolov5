
# Remove cached labels
rm /mnt/ssd2/Datasets/Fish_eye_dataset/PJ9/ROI_PJ9_person360/labels.cache

python train.py \
        --img 512 \
        --cfg ./models/yolov5m_toppan_2cls.yaml \
        --batch 16 \
        --epochs 50 \
        --data ./data/person_basket_2cls.yaml \
        --single-cls \
        --weights ./weights/yolov5m.pt\
        --device 2

