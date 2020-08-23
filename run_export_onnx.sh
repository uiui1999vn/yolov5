
export PYTHONPATH="$PWD"
python models/export.py \
       --weights weights/yolov5s.pt \
       --img 320 \
       --batch 1