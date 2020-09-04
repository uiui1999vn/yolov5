
# export PYTHONPATH=$PYTHONPATH:$PWD
export PYTHONPATH="$PWD"
python conversion/export.py \
       --weights weights/org/best_xs.pt \
       --img-size 320 \
       --batch-size 1 \
       --out-dir weights/onnx

