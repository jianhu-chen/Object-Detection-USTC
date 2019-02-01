if [ ! -d "object_detection/ssd_mobilenet/train_logs" ]; then
  mkdir -p object_detection/ssd_mobilenet/train_logs
fi

PIPELINE_CONFIG_PATH=object_detection/ssd_mobilenet/pipeline_ssd_mobilenet_v2_coco_2018_03_29.config
TRAIN_LOGS=object_detection/ssd_mobilenet/train_logs

python object_detection/legacy/train.py \
    --logtostderr \
    --train_dir=$TRAIN_LOGS \
    --pipeline_config_path=$PIPELINE_CONFIG_PATH
