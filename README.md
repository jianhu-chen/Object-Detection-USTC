# 使用TensorFlow Object Detection API 进行实时目标检测（基于SSD模型）

详细的介绍见博客：https://cjh.zone/2019/01/19/使用TensorFlow对象检测API进行实时目标检测/

## 目录结构

为了先对工程有个整体性的了解，故将此项目的目录结构列出如下：

```bash
Object-Detection-USTC
├── object_detection
│   ├── data # 存放数据
│   │   ├── mscoco_label_map.pbtxt # 预训练模型（coco数据集）的Label Maps
│   │   ├── pascal_label_map.pbtxt # 数据集2的Label Maps
│   │   ├── pascal_train.record # 数据集2生成的tfrecord格式的训练集
│   │   ├── pascal_val.record # 数据集2生成的tfrecord格式的验证集
│   │   ├── PennFudanPed # 数据集1（Penn-Fudan Database）
│   │   ├── PennFudanPed_label_map.pbtxt # 数据集1的Label Maps
│   │   ├── PennFudanPed_train.record # 数据集1生成的tfrecord格式的训练集
│   │   ├── PennFudanPed_val.record # 数据集1生成的tfrecord格式的验证集
│   │   └── VOC2007 # 数据集2
│   ├── dataset_tools # 数据集格式转换工具
│   │   ├── create_pascal_tf_record.py # 用于将本实验中的数据集2转换成tfrecord格式的脚本
│   │   └── ...# 用于将其他数据集转换成tfrecord格式的脚本文件
│   ├── legacy
│   │   ├── train.py # 用于训练我们自己的模型
│   │   └── ...
│   ├── ssd_mobilenet # 模型相关
│   │   ├── faster_rcnn_inception_v2_coco_2018_01_28 # 预训练模型１
│   │   ├── ssd_mobilenet_v1_coco_11_06_2017 # 预训练模型2
│   │   ├── ssd_mobilenet_v1_coco_2018_01_28 # 预训练模型3
│   │   ├── ssd_mobilenet_v2_coco_2018_03_29 # 预训练模型4（最终选用）
│   │   ├── output_inference_graph　# 导出的我们自己训练的模型
│   │   ├── pipeline_ssd_mobilenet_v2_coco_2018_03_29.config　# 管道配置文件
│   │   └── train_logs # 训练过程中产生的记录
│   │       ├── graph.pbtxt
│   │       ├── model.ckpt-1000.data-00000-of-00001
│   │       ├── model.ckpt-1000.index
│   │       ├── model.ckpt-1000.meta
│   │       └── ...
│   ├── export_inference_graph.py # 用于导出我们自己训练的模型的py脚本
│   ├── export_tflite_ssd_graph.py # 用于导出tflite压缩图的py脚本
│   └── ... # 其他文件略去
├── object_detection_video.py # 用于实时视频检测
├── utils # 实时视频检测时用到的两个库文件
│   ├── app_utils.py
│   └── test_app_utils.py
├── slim # 环境依赖
├── tflite # tflite产生的文件
│   ├── tflite_graph.pb
│   └── tflite_graph.pbtxt
├── create_pascal_tfrecord.sh # 用于将数据集2转换成tfrecord格式的shell脚本
├── train.sh # 用于执行训练命令的shell脚本
├── export_model.sh # 用于导出我们自己训练的模型的shell脚本
├── create_PennFudanPed_tfrecord.py # 将数据集1转换成tfrecord格式的py脚本
├── export_tflite_ssd_graph.sh # 用于导出tflite压缩图的shell脚本
└── video # 测试视频
    └── 2.mp4
```


