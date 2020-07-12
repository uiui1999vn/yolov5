
export PYTHONPATH="$PWD"
python models/onnx_export.py \
       --weights ./weights/ceot/last_ceot.pt \
       --img 640 640 \
       --batch 1
