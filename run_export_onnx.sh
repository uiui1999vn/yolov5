
# export PYTHONPATH=$PYTHONPATH:$PWD
export PYTHONPATH="$PWD"
python conversion/export.py \
       --weights weights/org/best.pt \
       --img-size 512 \
       --batch-size 1 \
       --out-dir weights/onnx

