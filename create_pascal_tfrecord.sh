python object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=object_detection/data  --year=VOC2007 --set=train \
    --output_path=object_detection/data/pascal_train.record

python object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=object_detection/data  --year=VOC2007 --set=val\
    --output_path=object_detection/data/pascal_val.record

