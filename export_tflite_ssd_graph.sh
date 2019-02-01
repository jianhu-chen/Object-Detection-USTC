if [ ! -d "tflite" ]; then
  mkdir -p tflite
fi

python object_detection/export_tflite_ssd_graph.py \
	--pipeline_config_path=/home/jhchen/Desktop/Object-Detection-USTC/object_detection/ssd_mobilenet/ssd_mobilenet_v1_coco_2018_01_28/pipeline.config \
	--trained_checkpoint_prefix=/home/jhchen/Desktop/Object-Detection-USTC/object_detection/ssd_mobilenet/ssd_mobilenet_v1_coco_2018_01_28/model.ckpt \
	--output_directory=tflite \
	--add_postprocessing_op=true

