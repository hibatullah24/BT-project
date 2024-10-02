import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tensorflow as tf
from tensorflow import keras








cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")





offset = 20
imgesize = 300


counter = 0

labels = ["A","B","C"]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgwhite = np.ones((imgesize, imgesize,3),np.uint8)*255
        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]

        imgCropShape = imgCrop.shape



        aspectRatio = h/w

        if aspectRatio >1:
            k = imgesize/h
            wcal = math.ceil(k*w)
            imgrsiza = cv2.resize(imgCrop,(wcal,imgesize))
            imgresizeShape = imgrsiza.shape
            wgap =math.ceil ((imgesize-wcal)/2)
            imgwhite[:, wgap:wcal+wgap] = imgrsiza
            prediction, index = classifier.getPrediction(img)
            print(prediction, index)



        else:
            k = imgesize / w
            hcal = math.ceil(k * h)
            imgrsiza = cv2.resize(imgCrop, (imgesize, hcal))
            imgresizeShape = imgrsiza.shape
            hgap = math.ceil((imgesize - hcal) / 2)
            imgwhite[ hgap:hcal + hgap, :] = imgrsiza


        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite", imgwhite)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

