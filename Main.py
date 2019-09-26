from matplotlib import pyplot as plt 
from tensorflow import keras

import os
import numpy as np
# Model reconstruction from JSON file

#Here i load the model
model = keras.models.load_model('model.h5')

imagenes = []
#I get the path to the image men.jpg in dir test_images
img  = os.path.join("test_images", "men.jpg")
#This is the object i have created to preprocess my images you should have yours
pre = Preprocess()
img_test = pre.prepare_image(img)
predictions = model.predict(img_test)
max = np.argmax(predictions)
cat = ["HOMBRE","MUJER"]
result = cat[max]
#result = "Men: "+str(predictions[0][0])+"% women: "+str(predictions[0][1])+"%"
plt.title(result)
printable = pre.visioned_image(img)
plt.imshow(printable)
    #plt.savefig(img+".png")

