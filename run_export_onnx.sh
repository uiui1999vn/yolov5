
export PYTHONPATH="$PWD"
python models/export.py \
       --weights weights/yolov5s.pt \
       --img 640 \
       --batch 1