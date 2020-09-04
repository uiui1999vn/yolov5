python ~/intel_2020R4/openvino/deployment_tools/model_optimizer/mo_onnx.py \
--input_model /Users/thanhnguyen/Documents/Sourcecodes/yolov5/weights/onnx/best_xs.onnx \
--output_dir /Users/thanhnguyen/Documents/Sourcecodes/yolov5/weights/openvino \
--input_shape="[1,3,320,320]" \
--log_level DEBUG

# --mean_values="[128, 128, 128]" \
# --scale_values="[255, 255, 255]" \