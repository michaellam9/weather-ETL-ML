import pandas as pd
import glob
from scipy import ndimage
import os.path
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.preprocessing import MinMaxScaler


targets = {'Mostly Cloudy' : 'Cloudy',
		'Mainly Clear' : 'Clear',
		'Rain Showers' : 'Rain',
		'Moderate Rain' : 'Rain'
	}

def changeLabel(s):
    try:
        return targets[s]
    except KeyError:
        return s

def extractDate(filename):
    name = os.path.basename(filename)
    name = name.split('-')[1]
    date = name[:4] + '-' + name[4:6] + '-' + name[6:8]
    time = name[8:10] + ':' + name[10:12]
    return date + ' ' + time

def img_to_nparray(path):
    img = np.array(Image.open(path))
    img = img.reshape(img.shape[0]*img.shape[1]*3)
    img = img/255
    return img

l = [pd.read_csv(filename, skiprows=16) for filename in glob.glob('yvr-weather/*.csv')]
data = pd.concat(l, axis=0).dropna(subset = ['Weather'])
data = data.drop(['Year','Month','Day','Time','Data Quality'], axis=1)
data['Date/Time'] = pd.to_datetime(data['Date/Time'])

images = [(extractDate(filename), filename) for filename in glob.glob('katkam-scaled/*.jpg')]
img_data = pd.DataFrame(images, columns=['Date/Time', 'Path'])
img_data['Date/Time'] = pd.to_datetime(img_data['Date/Time'])

final = data.merge(img_data, on=['Date/Time'])
final['Weather'] = final['Weather'].apply(changeLabel)


del data
del images
del img_data


sample = [final['Path'][0], final['Path'][1000]]
X = np.array([img_to_nparray(fname) for fname in final['Path']])
mu = np.mean(X, axis=0)
# print(X.shape)



# columns = np.array(range(250)).astype(str)
# images = pd.read_csv("cleaned_data.csv", usecols=columns)
# X = images.values
# del images




import pickle
pca = pickle.load(open("fitted_pca_model.sav", 'rb'))
# pickle code from: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

nComp = 100
Xhat = np.dot(pca.transform(X[2].reshape(1,-1))[:,:nComp], pca.components_[:nComp,:])
Xhat += mu
# above reverse pca code from: https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com/229093

print(Xhat[0,])


# img_orig = X[1]
# img_trans = pca.inverse_transform(pca.transform([img_orig]))[0]

# scaler = MinMaxScaler((0,255))
# img_trans = img_trans.reshape(-1,1)
# img_trans = scaler.fit_transform(img_trans)
img_trans = Xhat[0,].reshape([192,256,3])

# print(img_trans)

image = Image.fromarray(img_trans, 'RGB')
image.show()




# import pickle
# pickle.dump(pca, open("fitted_pca_model.sav", 'wb'))
# # pickle code from: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/