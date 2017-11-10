#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:44:19 2017

@author: ilies
"""

# Face recognition

# Import libraries
import cv2 as cv

# Load Cascades
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascade_eye.xml")

# Defining a function that will do the detection
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

# Doing some face recognition with webcam
video_capture = cv.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv.imshow('Video', canvas)
    if cv.waitKey(1) & 0xff == ord('q'):
        break
    
video_capture.release()
cv.destroyAllWindows()