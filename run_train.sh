:'
Analyzing anchors... anchors/target = 4.22, Best Possible Recall (BPR) = 0.9738. Attempting to generate improved anchors, please wait...
WARNING: Extremely small objects found. 7888 of 262465 labels are < 3 pixels in width or height.
Running kmeans for 9 anchors on 262175 points...
thr=0.25: 0.9881 best possible recall, 4.11 anchors past thr
n=9, img_size=320, metric_all=0.285/0.692-mean/best, past_thr=0.491-mean: 7,12,  17,30,  30,54,  46,90,  62,146,  125,104,  108,205,  278,121,  194,241
Evolving anchors with Genetic Algorithm: fitness = 0.7204: 100%|█████████████████████████████████| 1000/1000 [00:22<00:00, 44.66it/s]
thr=0.25: 0.9976 best possible recall, 4.33 anchors past thr
n=9, img_size=320, metric_all=0.296/0.720-mean/best, past_thr=0.489-mean: 4,8,  8,18,  16,27,  22,52,  50,64,  42,109,  86,154,  229,98,  166,212
New anchors saved to model. Update model *.yaml to use these anchors in the future.

Image sizes 320 train, 320 test
'

# Remove cached labels
rm  /home/thanh_nguyen/Datasets/COCO2017_Person/labels/*.cache
python train.py \
        --img 512 \
        --cfg ./models/yolov5x.yaml \
        --batch 12 \
        --epochs 200 \
        --data ./data/coco_person.yaml \
        --single-cls \
        --weights weights/yolov5x.pt \
        --device 0 \
        --multi-scale