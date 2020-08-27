:'
Analyzing anchors... anchors/target = 4.22, Best Possible Recall (BPR) = 0.9741. Attempting to generate improved anchors, please wait...
WARNING: Extremely small objects found. 7888 of 262465 labels are < 3 pixels in width or height.
Running kmeans for 9 anchors on 262175 points...
thr=0.25: 0.9882 best possible recall, 4.11 anchors past thr
n=9, img_size=320, metric_all=0.284/0.691-mean/best, past_thr=0.490-mean: 7,12,  17,30,  32,55,  46,93,  64,150,  120,99,  113,205,  276,119,  197,243
Evolving anchors with Genetic Algorithm: fitness = 0.7209: 100%|██████████████████████████████████████| 1000/1000 [00:13<00:00, 74.34it/s]
thr=0.25: 0.9959 best possible recall, 4.48 anchors past thr
n=9, img_size=320, metric_all=0.305/0.721-mean/best, past_thr=0.491-mean: 5,8,  8,18,  16,27,  23,53,  47,61,  41,113,  76,136,  177,110,  142,201
New anchors saved to model. Update model *.yaml to use these anchors in the future.

Image sizes 320 train, 320 test
'

# Remove cached labels
rm  /home/thanh_nguyen/Datasets/COCO2017_Person/labels/*.cache

python train.py \
        --img 512 \
        --cfg ./models/yolov5s.yaml \
        --batch 64 \
        --epochs 500 \
        --data ./data/coco_person.yaml \
        --single-cls \
        --weights ' ' \
        --device 0 \
        --multi-scale


