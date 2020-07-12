
#  python detect.py \
#         --weights ./weights/last.pt \
#         --img 640 \
#         --conf 0.4 \
#         --source ./inference/02_area1_shelf_left_2020_05_15_10_22_05.mp4 \
#         --device 1 \

EXP=exp0
IMG_SIZE=736
IMG_SET=/home/thanhnv/Datasets/sku110_yolo/images/test 

python detect.py \
        --weights ./runs/${EXP}/weights/best.pt \
        --img ${IMG_SIZE} \
        --conf 0.4 \
        --source ${IMG_SET} \
        --device 1 \