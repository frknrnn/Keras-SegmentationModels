import numpy as np
import tensorflow as tf
import cv2
import os
from Models.segmentationModels import SegmentationModels
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1

X = cv2.imread("../example.jpeg",0)

models = SegmentationModels()

images = models.crop_image(X,128,0,0)

model = models.FastBasicSegNet(IMG_WIDTH=IMG_WIDTH,IMG_HEIGHT=IMG_HEIGHT,IMG_CHANNELS=IMG_CHANNELS)
model.load_weights(".../model.hdf5")

X = np.array(images).reshape((len(images),128, 128, 1))

predictions = model.predict(X,verbose=1)

result = models.merge_image(predictions,128,128,3264,2448,0,0)

norm_image = cv2.normalize(result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F)
norm_image = norm_image.astype(np.uint8)
ret3, threshold = cv2.threshold(norm_image, 150, 255, cv2.THRESH_BINARY)


cv2.imwrite("segmentationOutput.png",threshold)
