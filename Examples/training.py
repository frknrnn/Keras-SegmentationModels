import tensorflow as tf
import time
from matplotlib import pyplot as plt
from Models.segmentationModels import SegmentationModels
from Dataset.createDataset import DataSet

t1=time.time()
# GPU
physical_devices = tf.config.list_physical_devices('GPU')
config = tf.config.experimental
config.set_memory_growth(physical_devices[0], True)
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1

modelName = "Res_Unet_115"

orginalPath = '.../orginal'
labelPath = '.../label'

dataset = DataSet()
X_train,Y_train=dataset.loadData(orginalPath,labelPath,True)

models = SegmentationModels()
model = models.Res_Unet(IMG_WIDTH=IMG_WIDTH,IMG_HEIGHT=IMG_HEIGHT,IMG_CHANNELS=IMG_CHANNELS)

results = model.fit(X_train, Y_train, batch_size=128,epochs=15,validation_split=0.2)#, callbacks=callbacks)


plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

t2=time.time()
print("Training time :"+str(t2-t1))

model.save_weights('../'+modelName+".hdf5")

