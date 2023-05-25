import numpy as np
import cv2
from   rknnlite.api import RKNNLite
from   yolov3_utils import yolov3_post_process, draw_image_boxes, download_yolov3_weight, show_top5, CLASSES, label_result_img
from   datetime import datetime
import sys


if __name__ == '__main__':

	MODEL_PATH = './yolov3.cfg'
	WEIGHT_PATH = './yolov3.weights'
	RKNN_MODEL_PATH = './yolov3_416.rknn'
	im_file = './data/dog_bike_car_416x416.jpg'
	DATASET = './data/dataset.txt'

	if (sys.argv[1]) : runs = int(sys.argv[1])
	else             : runs = 1
	
	rknn_model = RKNN_MODEL_PATH
	
	rknn_lite = RKNNLite(verbose=False)
	
	# Download yolov3.weight
	download_yolov3_weight(WEIGHT_PATH)

	# load RKNN model
	print('--> Load RKNN model')
	ret = rknn_lite.load_rknn(rknn_model)
	if ret != 0:
		print('Load RKNN model failed')
		exit(ret)
	print('done\n')

	#  load test image & resize to required 416 x 416 pixels expected by YOLO v3
	ori_img = cv2.imread(im_file)
	img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
	dim = (416,416)
	img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

	# init runtime environment
	print('--> Init runtime environment')
	ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
	#ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)     # : Running on NPU core 0.
	#ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)     # : Running on NPU core 1.
	#ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)     # : Running on NPU core 2.
	#ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1)   # ：Runing on NPU core 0 and core 1.
	#ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) # ：Runing on NPU core 0 and core 1 and core 2.
	#ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
	#rknn_lite = RKNNLite(verbose=True)
	#sdk_version = rknn_lite.get_sdk_version()
	#print("RKNN sdk_version ", sdk_version)
	print('done\n')

	# Inference
	print('--> Running model ', rknn_model, " ", runs, ' times')
	tstart = datetime.now()
	
	for i in range(runs): 
		outputs = rknn_lite.inference(inputs=[img]) # run inference using rknn network
		
	tstop = datetime.now()
	delta = tstop-tstart
	d = delta.total_seconds()
	it = d/runs
	fps = 1/it
	print("Completed", runs, "inferences @ %.3f" % (it*1000), "ms/inference (%.3f" % fps, "fps)\n")

	print('--> Label image with Inference results ')
	boxes, classes, scores = label_result_img(outputs, img, rknn_model, im_file)
	print('done\n')

	print('--> Inference results (top 3)')
	c_list = [CLASSES[index] for index in classes]
	for i in range(3): print(c_list[i], scores[i])
	#show_top5(rknn_model, outputs)
	print('done\n')
	

	rknn_lite.release()
