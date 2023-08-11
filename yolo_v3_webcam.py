#
# live opencv YOLO v3 inference based on example from 
# /rknn-toolkit2/examples/darknet/yolov3_416x416
# works with Ubuntu and USB webcam or OrangePi cam 13Mpixel based on ov13850
#

import cv2 # import opencv  for frame capture and image manipulation
from   rknnlite.api import RKNNLite # original example used RKNN API
from   yolov3_utils import yolov3_post_process, draw_image_boxes, download_yolov3_weight, show_top5, label_result_img, label_cv2_img, CLASSES


# setup for model
#
MODEL_PATH = './yolov3.cfg'
WEIGHT_PATH = './yolov3.weights'
RKNN_MODEL_PATH = './yolov3_416.rknn'
#
rknn_model = RKNN_MODEL_PATH
rknn_lite = RKNNLite(verbose=False) # using verbose option saves lots of state
download_yolov3_weight(WEIGHT_PATH) # download yolov3.weight from https://pjreddie.com/media/files/yolov3.weights

# load RKNN model
print('--> Load RKNN model')
ret = rknn_lite.load_rknn(rknn_model)
if ret != 0:
	print('failed to load RKNN model')
	exit(ret)
print('done\n')

# init runtime environment
print('--> Init runtime environment')
ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
if ret != 0:
	print('failed to initialise runtime environment')
	exit(ret)
print('done\n')

# set up camera capture
#
#dev_video=0 # USB webcam shows up as /dev/video0
dev_video=11 # OrangePi ov13850 cam1 shows up as /dev/video11
vid = cv2.VideoCapture(dev_video) 

# Set to VGA resolution as OrangePi cam defaults to 13MPixel
vid_w = 640
vid_h = 480
vid.set(3, vid_w) 
vid.set(4, vid_h)

# set up crop for inference RoI - 416x416 RoI out of VGA 640x480 frame
h=w=416 # crop is 416x416 square
left   = int((640-416)/2)
right  = left + w
bottom = int((480-416)/2)
top    = bottom + h

# define RoI box
lb = (left, bottom)
tr = (right, top)
color = (0, 0, 255) # Red color in BGR
tk = 2              # Line thickness of 2 px
#
font = cv2.FONT_HERSHEY_SIMPLEX
pos = (left+10,top-10)
fontScale = 0.75

print("\n\nYOLO v3 live inference on VGA frames (cropped to 416x416) using", len(CLASSES), "classes\n")
print(CLASSES)

# main image-capture and inference loop
#
while(True):
	ret, img = vid.read()                             # Capture the video frame by frame
	img = cv2.flip(img, 0)						      # flip image so it's less confusing to use
	roi = img[ bottom:top, left:right]                # crop image to 416 x 416 RoI required by YOLO v3
	outputs = rknn_lite.inference(inputs=[roi])       # run inference using rknn network	
	l_roi = label_cv2_img(outputs, roi)               # label RoI with inferences
	img[bottom:top, left:right] = l_roi               # fill labelled RoI back into the original frame for display
	img = cv2.rectangle(img, lb, tr, color, tk)       # frame RoI with red box
	img = cv2.putText(img, 'YOLO v3 RoI (416x416 pixels)', pos, font, fontScale, color, tk, cv2.LINE_AA)
	cv2.imshow('YOLOv3 inference (RoI 416x416)', img) # Display the resulting frame
	if cv2.waitKey(1) & 0xFF == ord('q'): break       # hit q to exit

# job done - now clean up before exit
#
rknn_lite.release()
vid.release()
cv2.destroyAllWindows()
