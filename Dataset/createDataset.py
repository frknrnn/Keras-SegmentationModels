import numpy as np
import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class DataSet:
    def loadData(self,imageFolderPath,labelFolderPath,augmentation,input_Shape=(128,128,1)):
        self.imageFolderPath = imageFolderPath
        self.labelFolderPath = labelFolderPath
        self.augflag = augmentation
        self.IMG_WIDTH = input_Shape[0]
        self.IMG_HEIGHT = input_Shape[1]
        self.IMG_CHANNELS = input_Shape[2]

        self.imageFileNames = os.listdir(imageFolderPath)
        self.labelFileNames = os.listdir(labelFolderPath)
        self.images = []
        self.masks = []
        self.getImageData()
        self.getMaskData()
        print(self.images[0])
        self.images = np.array(self.images).reshape((len(self.images),self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        self.masks = np.array(self.masks).reshape((len(self.masks),self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))

        return self.images,self.masks

    def getImageData(self):
        for i in self.imageFileNames:
            gray = cv2.imread(self.imageFolderPath + "/" + i, 0)
            if(self.augflag):
                subImages = self.augmentation(gray)
                self.images = self.images+subImages
            else:
                self.images.append(gray)

    def getMaskData(self):
        for i in self.labelFileNames:
            gray = cv2.imread(self.labelFolderPath + "/" + i, 0)
            ret, gray_threshold = cv2.threshold(gray, 150, 1, cv2.THRESH_BINARY)
            if(self.augflag):
                subImages = self.augmentation(gray_threshold)
                self.masks = self.masks + subImages
            else:
                self.masks.append(gray_threshold)

    def augmentation(self,image):
        sub = []
        sub.append(image)
        sub.append(cv2.flip(image, -1))
        sub.append(cv2.flip(image, 0))
        sub.append(cv2.flip(image, 1))
        sub.append(cv2.rotate(image,cv2.cv2.ROTATE_90_CLOCKWISE))
        sub.append(cv2.flip(sub[4], -1))
        sub.append(cv2.flip(sub[4], 0))
        sub.append(cv2.flip(sub[4], 1))
        return sub


