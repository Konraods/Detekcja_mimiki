from keras.models import load_model
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
import cv2


model = load_model('models\\anger_fear')

image = cv2.imread('input\\anger\\S011_004_00000020.png')
image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
img = np.array(image)
img = img / 255.0
img = img.reshape(1, 224, 224, 3)
label = model.predict(img)
if label <= 0.5:
    print("anger")
else:
    print("fear")
