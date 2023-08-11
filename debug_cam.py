# based on https://naghemanth.medium.com/camera-app-with-flask-and-opencv-bd147f6c0eec?source=friends_link&sk=705255bd58cf139ad95ab2149806d8c6)
#
from flask import Flask, render_template, Response, request
import cv2, datetime, time, os, sys
from threading import Thread
import numpy as np

# cat /usr/local/bin/test_camera.sh
#
#if [[ -c /dev/video51 ]]; then
#        gst-launch-1.0 v4l2src device=/dev/video33 io-mode=4 ! video/x-raw,format=NV12,width=720,height=576,framerate=15/1 ! xvimagesink > /dev/null 2>&1 &
#        gst-launch-1.0 v4l2src device=/dev/video42 io-mode=4 ! video/x-raw,format=NV12,width=720,height=576,framerate=15/1 ! xvimagesink > /dev/null 2>&1 &
#        gst-launch-1.0 v4l2src device=/dev/video51 io-mode=4 ! video/x-raw,format=NV12,width=720,height=576,framerate=15/1 ! xvimagesink > /dev/null 2>&1
#elif [[ -c /dev/video31 ]]; then
#        gst-launch-1.0 v4l2src device=/dev/video22 io-mode=4 ! video/x-raw,format=NV12,width=720,height=576,framerate=15/1 ! xvimagesink > /dev/null 2>&1 &
#        gst-launch-1.0 v4l2src device=/dev/video31 io-mode=4 ! video/x-raw,format=NV12,width=720,height=576,framerate=15/1 ! xvimagesink > /dev/null 2>&1
#elif [[ -c /dev/video11 ]]; then
#        gst-launch-1.0 v4l2src device=/dev/video11 io-mode=4 ! video/x-raw,format=NV12,width=720,height=576,framerate=15/1 ! xvimagesink > /dev/null 2>&1

video_dev = int(sys.argv[1])
cap = cv2.VideoCapture(video_dev)

while True:
    ret, frame = cap.read()
    cv2.imshow('MIPI cam0',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
