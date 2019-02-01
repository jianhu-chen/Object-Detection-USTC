# -*- coding: utf-8 -*-
# @File 	: object_detection_video.py
# @Author 	: jianhuChen
# @Date 	: 2018-12-23 16:53:11
# @License 	: Copyright(C), USTC
# @Last Modified by  : jianhuChen
# @Last Modified time: 2019-01-18 18:26:44
# 
import tensorflow as tf
import cv2 # opencv库
import os
import time
import numpy as np
from queue import Queue #　队列
from threading import Thread　# 线程
from object_detection.utils import label_map_util # tf官方api，用于读取label maps
from utils.app_utils import draw_boxes_and_labels, FPS # 用于解析图片识别结果，记录FPS

# 视频尺寸，不用做更改，后面会有代码自动获取并更新此值
width = 640
height = 480

MODEL1_FLAG = True
MODEL2_FLAG = True

CWD_PATH = os.getcwd()

if MODEL1_FLAG:
	# 模型名字
	MODEL_NAME = 'output_inference_graph-1000'
	# 模型的checkpoint文件路径
	PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', 'ssd_mobilenet', MODEL_NAME, 'frozen_inference_graph.pb')
	# 模型的label map路径
	PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'pascal_label_map.pbtxt')
	#　模型能检测的对象类别数
	NUM_CLASSES = 4
	# 调用官方库加载label map, 用于识别出物品后展示类别名
	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	# 将读入的map文件转换成字典的列表,如：[{'name': 'person', 'id': 1}, ...]
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	# 得到每个类别的id和name，字典，{1: {'id': 1, 'name': 'person'}, 2: {'id': 2, 'name': 'bicycle'}}
	category_index = label_map_util.create_category_index(categories)

if MODEL2_FLAG:
	MODEL_NAME2 = 'ssd_mobilenet_v1_coco_2018_01_28'  # fps: 18.88
	# MODEL_NAME2 = 'ssd_mobilenet_v2_coco_2018_03_29' # 14.33
	# MODEL_NAME2 = 'ssd_mobilenet_v1_coco_11_06_2017' # 14.57
	# MODEL_NAME2 = 'faster_rcnn_inception_v2_coco_2018_01_28' # 4.92	PATH_TO_CKPT2 = os.path.join(CWD_PATH, 'object_detection', 'ssd_mobilenet', MODEL_NAME2, 'frozen_inference_graph.pb')
	PATH_TO_LABELS2 = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
	NUM_CLASSES2 = 90
	label_map2 = label_map_util.load_labelmap(PATH_TO_LABELS2)
	categories2 = label_map_util.convert_label_map_to_categories(label_map2, max_num_classes=NUM_CLASSES2, use_display_name=True)
	category_index2 = label_map_util.create_category_index(categories2)

def detect_objects(image_np, sess, detection_graph, category_index):
	# 扩展图像的尺寸, 因为模型要求图像具有形状: [1, None, None, 3]
	image_np_expanded = np.expand_dims(image_np, axis=0)

	image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	# 检测结果，每个框表示检测到特定对象的图像的一部分
	boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	# scores表示识别出的物体的置信度 是一个概率值，他最后会与label一起显示在图像上
	scores = detection_graph.get_tensor_by_name('detection_scores:0')
	classes = detection_graph.get_tensor_by_name('detection_classes:0')
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')

	# 运行会话开始检测并返回结果，feed为传入的图像张量
	(boxes, scores, classes, num_detections) = sess.run(
		[boxes, scores, classes, num_detections],
		feed_dict={image_tensor: image_np_expanded})

	# 检测结果的可视化，也就是将返回的框框画在图像上
	# 返回框坐标、类名和颜色
	rect_points, class_names, class_colors = draw_boxes_and_labels(
		boxes=np.squeeze(boxes),
		classes=np.squeeze(classes).astype(np.int32),
		scores=np.squeeze(scores),
		category_index=category_index,
		min_score_thresh=.3  # 可视化的最低分数阈值
	)

	return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)

# 调整亮度和对比度
# c:对比度, b:亮度
def contrast_brightness_image(img, c, b):
	h, w, ch = img.shape  # 获取shape的数值，height/width/channel
	# 新建全零图片数组blank,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
	blank = np.zeros([h, w, ch], img.dtype)
	dst = cv2.addWeighted(img, c, blank, 1-c, b) # 计算两个图像阵列的加权和 dst=src1*alpha+src2*beta+gamma
	return dst

# 用于对象检测的线程
def thread_worker(input_q, output_q):
	# Load a (frozen) Tensorflow model into memory.
	if MODEL1_FLAG:
		detection_graph = tf.Graph()
		with detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
				sess = tf.Session(graph=detection_graph)

	if MODEL2_FLAG:
		detection_graph2 = tf.Graph()
		with detection_graph2.as_default():
			od_graph_def2 = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_CKPT2, 'rb') as fid2:
				serialized_graph2 = fid2.read()
				od_graph_def2.ParseFromString(serialized_graph2)
				tf.import_graph_def(od_graph_def2, name='')
				sess2 = tf.Session(graph=detection_graph2)
	# 设置奇数检测，偶数不检测，第一帧一定要检测，所以设置初始值为True
	detect_flag = True
	while True:
		# 从输入队列里取出待检测的图片
		frame = input_q.get()
		# 由于使用opencv读入的图片是BGR色彩模式，我们需要将其转换成RGB模式
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# 图像预处理，增强对比度和亮度
		con_bri_frame = contrast_brightness_image(frame, 1.3, 0)
		# 设置在图片上标注label时的字体
		font = cv2.FONT_HERSHEY_SIMPLEX
		if MODEL1_FLAG:
			if detect_flag: # 判断是否需要检测，如果为False，后面的代码会使用上一次检测出的data来标注
				# 调用目标检测函数，需要传入预处理后的图像，tf会话，图，
				# 返回的是一个字典，其中包含的信息有：对象边界的两个点（左上角和右下角）,对象的类名, 标注此对象时的框框颜色
				data1 = detect_objects(con_bri_frame, sess, detection_graph, category_index)
			rec_points = data1['rect_points']  # 获取BBOX的坐标
			class_names = data1['class_names']  # 获取类名
			# print(class_names)
			class_colors = data1['class_colors']
			for point, name, color in zip(rec_points, class_names, class_colors):
				# 获得检测出的对象信息：左上角顶点和右下角顶点的坐标 类名 颜色
				# 在图片上标注这些检测出来的对象
				if name[0].startswith('face') or name[0].startswith('person'):
					cv2.rectangle(frame, (int(point['xmin'] * width), int(point['ymin'] * height)), (int(point['xmax'] * width), int(point['ymax'] * height)), color, 2)
					cv2.rectangle(frame, (int(point['xmin'] * width), int(point['ymin'] * height)), (int(point['xmin'] * width) + len(name[0]) * 6, int(point['ymin'] * height) - 10), color, -1, cv2.LINE_AA)
					cv2.putText(frame, name[0], (int(point['xmin'] * width), int(point['ymin'] * height)), font, 0.3, (0, 0, 0), 1)

		if MODEL2_FLAG:
			if detect_flag:
				data2 = detect_objects(con_bri_frame, sess2, detection_graph2, category_index2)
			rec_points = data2['rect_points']
			class_names = data2['class_names']
			class_colors = data2['class_colors']
			for point, name, color in zip(rec_points, class_names, class_colors):
				if name[0].startswith('traffic light') or (MODEL1_FLAG==False and (name[0].startswith('person'))):
					cv2.rectangle(frame, (int(point['xmin'] * width), int(point['ymin'] * height)), (int(point['xmax'] * width), int(point['ymax'] * height)), color, 2)
					cv2.rectangle(frame, (int(point['xmin'] * width), int(point['ymin'] * height)), (int(point['xmin'] * width) + len(name[0]) * 6, int(point['ymin'] * height) - 10), color, -1, cv2.LINE_AA)
					cv2.putText(frame, name[0], (int(point['xmin'] * width), int(point['ymin'] * height)), font, 0.3, (0, 0, 0), 1)

		# 交替检测
		detect_flag = not detect_flag
		# 将检测完并标注好对象信息的图片塞入输出队列
		output_q.put(frame)
	if MODEL1_FLAG:
		sess.close() # 关闭会话
	if MODEL2_FLAG:
		sess2.close()

def main():
	speed = 20 # 视频速度控制
	thread_num = 1 # 线程数量
	get_frame_num = 1 #　已经显示图片的数量
	video_path = "video/2.mp4" # 待检测的视频路径
	input_q = [Queue(400), # 输入队列列表，容量为400
				# Queue(400),
				# Queue(400),
				] 
	output_q = [Queue(), # 输出队列列表，无限大容量
				# Queue(),
				# Queue(),
				]
	for i in range(thread_num): # 进程的个数
		t = Thread(target=thread_worker, args=(input_q[i], output_q[i]))
		t.daemon = True # 这个线程是不重要的，在进程退出的时候，不用等待这个线程退出
		t.start()

	# 开始读取视频
	video_capture = cv2.VideoCapture(video_path)  # 导入视频
	global width, height # 通过opencv获取视频的尺寸
	width, height = int(video_capture.get(3)), int(video_capture.get(4))
	print('video width-height:', width, '-',height)
	fps = FPS().start() # 开始计算FPS，这句话的作用是打开计时器开始计时
	while True:
		ret, frame = video_capture.read() # 读取视频帧
		if ret == False: # 读完图片退出
			break
		fps.update() # 每读一帧，计数+1
		# if not input_q.full():
		in_q_index = fps.getNumFrames()%thread_num # 计算该帧图片应该入哪个输入队列
		input_q[in_q_index].put(frame) # 将该帧图片入输入队列

		frame_start_time = time.time() # 计录处理当前帧图片的起始时间
		out_q_index = get_frame_num%thread_num # 计算目前应该从哪个输出队列取图片显示
		if not output_q[out_q_index].empty():
			get_frame_num += 1 # 已经显示的图片数量+1
			# 将从输出队列获取到的图片色彩模式转换回BGR，再显示
			od_frame = cv2.cvtColor(output_q[out_q_index].get(), cv2.COLOR_RGB2BGR)
			ch = cv2.waitKey(speed) # 检测按键
			if ch & 0xFF == ord('q'): # q键：退出
				break
			elif ch & 0xFF == ord('w'): # w键：速度减慢
				speed += 10
			elif ch & 0xFF == ord('s'): # s键：速度加快
				speed -= 10
			elif ch & 0xFF == ord('r'): # r键：恢复初始速度
				speed = 50
			# 将速度放到图片左上角去
			cv2.putText(od_frame, 'SPEED:' + str(speed), (20, int(height/20)), cv2.FONT_HERSHEY_SIMPLEX,
						0.7, (0, 255, 0), 2)
			# 将当前帧数、运行时间、平均帧率标注到图片左上角去
			fps.stop()
			cv2.putText(od_frame, 'FRAME:{:}'.format(fps._numFrames), (20, int(height*2/20)), cv2.FONT_HERSHEY_SIMPLEX,
						0.8, (0, 255, 0), 2)
			cv2.putText(od_frame, 'TIME:{:.3f}'.format(fps.elapsed()), (20, int(height*3/20)), cv2.FONT_HERSHEY_SIMPLEX,
						0.8, (0, 255, 0), 2)
			cv2.putText(od_frame, 'AVE_FPS: {:.3f}'.format(fps.fps()), (20, int(height*4/20)), cv2.FONT_HERSHEY_SIMPLEX,
						0.7, (0, 0, 255), 2)
			cv2.imshow('Video', od_frame)
		# 打印当前帧处理所花的时间
		print('[INFO] elapsed time: {:.5f}'.format(time.time() - frame_start_time))

	fps.stop()
	# 打印总时间
	print('[INFO] elapsed time (total): {:.4f}'.format(fps.elapsed()))
	#　打印平均帧率
	print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()