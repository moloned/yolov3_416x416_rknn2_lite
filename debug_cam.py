# based on https://naghemanth.medium.com/camera-app-with-flask-and-opencv-bd147f6c0eec?source=friends_link&sk=705255bd58cf139ad95ab2149806d8c6)
#
from flask import Flask, render_template, Response, request
import cv2, datetime, time, os, sys
from threading import Thread
import numpy as np

import cv2

cap = cv2.VideoCapture(11)

while True:

    ret, frame = cap.read()
    cv2.imshow('MIPI cam0',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
