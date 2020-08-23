
export PYTHONPATH="$PWD"
python models/export.py \
       --weights weights/best.pt \
       --img 640 \
       --batch 1