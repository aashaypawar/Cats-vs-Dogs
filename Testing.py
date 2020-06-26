from tensorflow import keras
from keras.preprocessing import image
import numpy as np

model = keras.models.load_model('output.h5')

img_pred = image.load_img('test/12481.jpg',target_size=(150,150))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred,axis = 0)

result = model.predict(img_pred)

if(result[0][0] == 1):
	print("Dog")
else:
	print("Cat")