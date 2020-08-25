
export PYTHONPATH="$PWD"
python models/export.py \
       --weights weights/best.pt \
       --img 320 \
       --batch 1