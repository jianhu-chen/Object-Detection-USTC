if [ ! -d "object_detection/ssd_mobilenet/output_inference_graph" ]; then
  mkdir -p object_detection/ssd_mobilenet/output_inference_graph
fi


python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path object_detection/ssd_mobilenet/pipeline_ssd_mobilenet_v2_coco_2018_03_29.config \
    --trained_checkpoint_prefix object_detection/ssd_mobilenet/train_logs/model.ckpt-500 \
    --output_directory object_detection/ssd_mobilenet/output_inference_graph

