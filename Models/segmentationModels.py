import tensorflow as tf
import os
import cv2
import numpy as np
from keras.layers import Input,concatenate,MaxPooling2D,Conv2DTranspose
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from linknet import LinkNet

class SegmentationModels:

    def dataAugmentation(self):
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
        ])
        return data_augmentation

    def crop_image(self,im, unet_input_size, shift_x, shift_y):
        size_im = im.shape
        #im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        pad_size_x, pad_size_y = 0, 0
        if size_im[0] % unet_input_size != 0:
            pad_size_y = (size_im[0] // unet_input_size + 1) * unet_input_size - size_im[0]
        if size_im[1] % unet_input_size != 0:
            pad_size_x = (size_im[1] // unet_input_size + 1) * unet_input_size - size_im[1]

        pad_im = np.pad(im, ((shift_y, pad_size_y + shift_y), (shift_x, pad_size_x + shift_x)),
                        mode='mean')  # )mode='constant',constant_values=0
        size_pad_im = pad_im.shape
        x_div, y_div = np.int64(size_pad_im[1] / unet_input_size), np.int64(size_pad_im[0] / unet_input_size)

        im_cropped = []
        count = 0
        for j in range(y_div):
            for i in range(x_div):
                cropped = pad_im[j * unet_input_size:(j + 1) * unet_input_size,
                          i * unet_input_size:(i + 1) * unet_input_size]
                im_cropped.append(np.uint8(cropped))
                count = count + 1

        return im_cropped

    def merge_image(self, images, cr_sizeX, cr_sizeY, fullResX, fullResY, shift_x, shift_y):
        v_images = []
        add_y = 0
        if (fullResY % 128 != 0):
            add_y = 1
        y_scan = fullResY // cr_sizeY + add_y
        add_x = 0
        if (fullResX % 128 != 0):
            add_x = 1
        x_scan = fullResX // cr_sizeX + add_x
        for i in range(y_scan):
            v_img = cv2.hconcat(images[(i * x_scan):((i * x_scan) + x_scan)])
            v_images.append(v_img)
        image_full = cv2.vconcat(v_images)

        return image_full[shift_y:fullResY + shift_y, shift_x:fullResX + shift_x]


    def UNet(self,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS):
        # Build the model
        inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

        # Contraction path
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.Dropout(0.3)(c5)
        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Expansive path
        u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        model.summary()
        return model

    def UNet_Simple(self,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS):
        # Build the model
        inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

        # Contraction path
        c1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.Dropout(0.3)(c5)
        c5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Expansive path
        u6 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        c7 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        model.summary()
        return model

    def UNet_Basic(self,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS):
        # Build the model
        inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

        # Contraction path
        c1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)


        # Expansive path

        u8 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c3)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        model.summary()
        return model

    def FastBasicSegNet(self,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS):
        # Build the model
        inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

        # Contraction path
        c1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)

        # Expansive path

        u9 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c2)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        model.summary()
        return model

    def Segnet(self,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS, kernel=3, pool_size=(2, 2), output_mode="softmax"):
        # encoder
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
        conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(s)
        conv_1 = BatchNormalization()(conv_1)
        conv_1 = Activation("relu")(conv_1)
        conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
        conv_2 = BatchNormalization()(conv_2)
        conv_2 = Activation("relu")(conv_2)

        pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

        conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
        conv_3 = BatchNormalization()(conv_3)
        conv_3 = Activation("relu")(conv_3)
        conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
        conv_4 = BatchNormalization()(conv_4)
        conv_4 = Activation("relu")(conv_4)

        pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

        conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
        conv_5 = BatchNormalization()(conv_5)
        conv_5 = Activation("relu")(conv_5)
        conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
        conv_6 = BatchNormalization()(conv_6)
        conv_6 = Activation("relu")(conv_6)
        conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
        conv_7 = BatchNormalization()(conv_7)
        conv_7 = Activation("relu")(conv_7)

        pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

        conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
        conv_8 = BatchNormalization()(conv_8)
        conv_8 = Activation("relu")(conv_8)
        conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
        conv_9 = BatchNormalization()(conv_9)
        conv_9 = Activation("relu")(conv_9)
        conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
        conv_10 = BatchNormalization()(conv_10)
        conv_10 = Activation("relu")(conv_10)

        pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

        conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
        conv_11 = BatchNormalization()(conv_11)
        conv_11 = Activation("relu")(conv_11)
        conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
        conv_12 = BatchNormalization()(conv_12)
        conv_12 = Activation("relu")(conv_12)
        conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
        conv_13 = BatchNormalization()(conv_13)
        conv_13 = Activation("relu")(conv_13)

        pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
        print("Build enceder done..")

        # decoder

        unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

        conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
        conv_14 = BatchNormalization()(conv_14)
        conv_14 = Activation("relu")(conv_14)
        conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
        conv_15 = BatchNormalization()(conv_15)
        conv_15 = Activation("relu")(conv_15)
        conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
        conv_16 = BatchNormalization()(conv_16)
        conv_16 = Activation("relu")(conv_16)

        unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

        conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
        conv_17 = BatchNormalization()(conv_17)
        conv_17 = Activation("relu")(conv_17)
        conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
        conv_18 = BatchNormalization()(conv_18)
        conv_18 = Activation("relu")(conv_18)
        conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
        conv_19 = BatchNormalization()(conv_19)
        conv_19 = Activation("relu")(conv_19)

        unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

        conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
        conv_20 = BatchNormalization()(conv_20)
        conv_20 = Activation("relu")(conv_20)
        conv_21 = Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
        conv_21 = BatchNormalization()(conv_21)
        conv_21 = Activation("relu")(conv_21)
        conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
        conv_22 = BatchNormalization()(conv_22)
        conv_22 = Activation("relu")(conv_22)

        unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

        conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
        conv_23 = BatchNormalization()(conv_23)
        conv_23 = Activation("relu")(conv_23)
        conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
        conv_24 = BatchNormalization()(conv_24)
        conv_24 = Activation("relu")(conv_24)

        unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

        conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
        conv_25 = BatchNormalization()(conv_25)
        conv_25 = Activation("relu")(conv_25)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv_25)

        print("Build decoder done..")

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        model.summary()

        return model

    def Res_Unet(self,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS):
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
        conv1 = Convolution2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Convolution2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conc1 = concatenate([inputs, conv1], axis=3)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conc1)

        conv2 = Convolution2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Convolution2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conc2 = concatenate([pool1, conv2], axis=3)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conc2)

        conv3 = Convolution2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Convolution2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conc3 = concatenate([pool2, conv3], axis=3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conc3)

        conv4 = Convolution2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Convolution2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conc4 = concatenate([pool3, conv4], axis=3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conc4)

        conv5 = Convolution2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Convolution2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conc5 = concatenate([pool4, conv5], axis=3)

        up6 = concatenate(
            [Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conc5),
             conv4], axis=3)
        conv6 = Convolution2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
        conv6 = Convolution2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        conc6 = concatenate([up6, conv6], axis=3)

        up7 = concatenate(
            [Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conc6),
             conv3], axis=3)
        conv7 = Convolution2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up7)
        conv7 = Convolution2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        conc7 = concatenate([up7, conv7], axis=3)

        up8 = concatenate(
            [Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conc7), conv2],
            axis=3)
        conv8 = Convolution2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up8)
        conv8 = Convolution2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conc8 = concatenate([up8, conv8], axis=3)

        up9 = concatenate(
            [Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conc8), conv1],
            axis=3)
        conv9 = Convolution2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up9)
        conv9 = Convolution2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conc9 = concatenate([up9, conv9], axis=3)

        outputs = Convolution2D(1, (1, 1), activation='sigmoid')(conc9)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])

        return model


    def Linknet(self,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS):
        model = LinkNet((IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS))
        model.summary()
        return model
