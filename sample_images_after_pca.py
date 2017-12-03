import pandas as pd
import glob
from scipy import ndimage
import os.path
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image

columns = np.array(range(250)).astype(str)
# print(columns)
images = pd.read_csv("cleaned_data.csv", usecols=columns)
X = images.values
del images

X = X*255

def img_to_nparray(path):
    img = np.array(Image.open(path))
    return img

X = np.array([img_to_nparray(fname) for fname in final['Path']])


import pickle
pca = pickle.load(open("fitted_pca_model.sav", 'rb'))
# pickle code from: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

img_orig = X[0]
img_trans = pca.inverse_transform([img_orig])[0]
img_trans = img_trans.reshape([192,256,3])

image = Image.fromarray(img_trans, 'RGB')
image.show()




# import pickle
# pickle.dump(pca, open("fitted_pca_model.sav", 'wb'))
# # pickle code from: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/