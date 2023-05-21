import cv2
import dlib
import requests
import pyttsx3
from scipy.spatial import distance

import time
import sys
import ibmiotf.application
import ibmiotf.device
import keyboard
import random

organization = "a922ub"
deviceType = "Detector"
deviceId = "8402"
authMethod = "token"
authToken = "8667370445"

def ibmstart(x):
    def myCommandCallback(cmd):
        print("Command received: %s" % cmd.data['command'])
        print(cmd)

    try:
      deviceOptions = {"org": organization, "type": deviceType, "id": deviceId, "auth-method": authMethod, "auth-token": authToken}
      deviceCli = ibmiotf.device.Client(deviceOptions)
    except Exception as e:
      print("Caught exception connecting device: %s" % str(e))
      sys.exit()
    deviceCli.connect()  
    data = { 'Status' : x}
    print(data)
    def myOnPublishCallback():
        print ("Published Status = %s" % x, "to IBM Watson")

    success = deviceCli.publishEvent("DD", "json", data, qos=0, on_publish=myOnPublishCallback)
    if not success:
        print("Not connected to IoTF")
    deviceCli.commandCallback = myCommandCallback
    deviceCli.disconnect()

engine = pyttsx3.init()
cap = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
# dlib_facelandmark = dlib.shape_predictor("https://drive.google.com/file/d/1LLdF78rqFAvdL7hjLLALOSdoWa7cYCcR/view?usp=share_link")
# dlib_facelandmark = dlib.shape_predictor("https://drive.google.com/uc?export=view&id=1LLdF78rqFAvdL7hjLLALOSdoWa7cYCcR")
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def Detect_Eye(eye):
  poi_A = distance.euclidean(eye[1], eye[5])
  poi_B = distance.euclidean(eye[2], eye[4])
  poi_C = distance.euclidean(eye[0], eye[3])
  aspect_ratio_Eye = (poi_A+poi_B)/(2*poi_C)
  return aspect_ratio_Eye

while True:
  null, frame = cap.read()
  flag=0
  gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_detector(gray_scale)
  for face in faces:
    face_landmarks = dlib_facelandmark(gray_scale, face)
    leftEye = []
    rightEye = []
    for n in range(42, 48):
      x = face_landmarks.part(n).x
      y = face_landmarks.part(n).y
      rightEye.append((x, y))
      next_point = n+1
      if n == 47:
        next_point = 42
      x2 = face_landmarks.part(next_point).x
      y2 = face_landmarks.part(next_point).y
      cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)
    for n in range(36, 42):
      x = face_landmarks.part(n).x
      y = face_landmarks.part(n).y
      leftEye.append((x, y))
      next_point = n+1
      if n == 41:
        next_point = 36
      x2 = face_landmarks.part(next_point).x
      y2 = face_landmarks.part(next_point).y
      cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)
    right_Eye = Detect_Eye(rightEye)
    left_Eye = Detect_Eye(leftEye)
    Eye_Rat = (left_Eye+right_Eye)/2
    Eye_Rat = round(Eye_Rat, 2)
    if Eye_Rat < 0.25:
      cv2.putText(frame, "DROWSINESS DETECTED", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
      cv2.putText(frame, "Alert!!!! WAKE UP DUDE", (50, 450), cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)
      engine.say("WAKE UP")
      flag=1
      engine.runAndWait()

  cv2.imshow("Drowsiness Detector", frame)
  print(flag)
  ibmstart(flag)
  key = cv2.waitKey(1)
  if (keyboard.is_pressed("q")):
    break

cap.release()
cv2.destroyAllWindows()
