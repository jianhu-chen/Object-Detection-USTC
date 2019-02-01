# -*- coding: utf-8 -*-
# @File 	: create_PennFudanPed_tfrecord.py
# @Author 	: jianhuChen
# @Date 	: 2018-12-23 16:53:11
# @License 	: Copyright(C), USTC
# @Last Modified by  : jianhuChen
# @Last Modified time: 2019-01-17 17:54:31
import os
import io
import pandas as pd
import numpy as np
import tensorflow as tf

from object_detection.utils import dataset_util

from tqdm import *

def data_pre_processing(annotation_path):
	datas = [] # 存储图片信息
	files = os.listdir(annotation_path)
	for file in files:
		sample = {}
		xmins = []
		xmaxs = []
		ymins = []
		ymaxs = []
		classes_text = []
		classes = []
		with open(annotation_path+'/'+file, 'r') as fp:
			lines = fp.readlines()
			for line in lines:
				if line.startswith('Image filename :'):
					filename = line.strip().split('/')[-1][:-1] # get image name
					image_format = filename.split('.')[-1] # get image format
				elif line.startswith('Image size (X x Y x C) :'):
					width_height_channel = line.split(':')[-1].split('x')
					width = eval(width_height_channel[0].strip()) # get image width
					height = eval(width_height_channel[1].strip()) # get image height
				elif line.startswith('Original label for object'):
					label = line.split(':')[-1].strip()[1:-1]
					if label == 'PennFudanPed': # get the object id
						classes.append(1)
						classes_text.append('person'.encode('utf8')) # get the object label
				elif line.startswith('Bounding box for object'):
					Xmin_Ymin_Xmax_Ymax = line.split(':')[-1].strip() # like:(160, 182) - (302, 431)
					xmin = eval(Xmin_Ymin_Xmax_Ymax.split(',')[0][1:])
					ymin = eval(Xmin_Ymin_Xmax_Ymax.split(',')[1].partition(') - (')[0])
					xmax = eval(Xmin_Ymin_Xmax_Ymax.split(',')[1].partition(') - (')[2])
					ymax = eval(Xmin_Ymin_Xmax_Ymax.split(',')[2][:-1])
					xmins.append(xmin)
					xmaxs.append(xmax)
					ymins.append(ymin)
					ymaxs.append(ymax)
			# 将该图片所有属性添加到字典里   
			sample['filename'] = filename
			sample['image_format'] = image_format
			sample['width'] = width
			sample['height'] = height
			sample['classes_text'] = classes_text
			sample['classes'] = classes
			sample['xmins'] = xmins
			sample['xmaxs'] = xmaxs
			sample['ymins'] = ymins
			sample['ymaxs'] = ymaxs
			# 将该图片添加到数据集列表里
			datas.append(sample)
	return datas


def create_tf_example(sample, image_path):
	height = sample['height']
	width = sample['width']
	filename = sample['filename']
	image_format = sample['image_format']
	# 注意：坐标需要标准化
	xmins = np.array(sample['xmins'])/width
	xmaxs = np.array(sample['xmaxs'])/width
	ymins = np.array(sample['ymins'])/height
	ymaxs = np.array(sample['ymaxs'])/height
	classes_text = sample['classes_text']
	classes = sample['classes']

	# 读取图片数据
	encoded_img = tf.gfile.FastGFile(image_path+'/'+sample['filename'],'rb').read()
	
	tf_example = tf.train.Example(features=tf.train.Features(feature={
		'image/height': dataset_util.int64_feature(height),
		'image/width': dataset_util.int64_feature(width),
		'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
		'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
		'image/encoded': dataset_util.bytes_feature(encoded_img),
		'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
		'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
		'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
		'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
		'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
		'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
		'image/object/class/label': dataset_util.int64_list_feature(classes),
	}))

	return tf_example
 

def main(_):
	# 标注文件路径
	annotation_path = r'object_detection/data/PennFudanPed/Annotation'
	# 图片文件路径
	image_path = r'object_detection/data/PennFudanPed/PNGImages'
	# tfrecord输出路径
	output_path = r'object_detection/data'

	train_writer = tf.python_io.TFRecordWriter(os.path.join(output_path, 'PennFudanPed_train.record'))
	val_writer = tf.python_io.TFRecordWriter(os.path.join(output_path, 'PennFudanPed_val.record'))

	annotation_datas = data_pre_processing(annotation_path)
	for sample in tqdm(annotation_datas[:]):
		tf_sample = create_tf_example(sample, image_path)
		train_writer.write(tf_sample.SerializeToString())
	print('Successfully created the TFRecords(train).')

	for sample in tqdm(annotation_datas[150:]):
		tf_sample = create_tf_example(sample, image_path)
		val_writer.write(tf_sample.SerializeToString())
	print('Successfully created the TFRecords(val).')

	train_writer.close()
	val_writer.close()
 
if __name__ == '__main__':
	tf.app.run()

