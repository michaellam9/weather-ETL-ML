import pandas as pd
import glob
from scipy import ndimage
import os.path
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt


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

## Reading Images and doing PCA to reduce image features
X = np.array([img_to_nparray(fname) for fname in final['Path']])

index1 = final['Path'][final['Path'] == "katkam-scaled/katkam-20160605070000.jpg"].index[0]
original_image1 = X[index1].reshape([192,256,3])
index2 = final['Path'][final['Path'] == "katkam-scaled/katkam-20171031130000.jpg"].index[0]
original_image2 = X[index2].reshape([192,256,3])

pca = PCA(250)
X = pca.fit_transform(X)
variance = pca.explained_variance_ratio_
total = 0
for i in range(len(variance)):
    total = total + variance[i]
print("Total explained variance ratio with 250 features: " + str(total))

transformed_image1 = pca.inverse_transform(X[index1]).reshape([192,256,3])
plt.imshow(original_image1)
plt.savefig("pre_transform_clear_sky.png")

plt.imshow(transformed_image1)
plt.savefig("post_transform_clear_sky.png")

transformed_image2 = pca.inverse_transform(X[index2]).reshape([192,256,3])
plt.imshow(original_image2)
plt.savefig("pre_transform_cloudy_sky.png")

plt.imshow(transformed_image2)
plt.savefig("post_transform_cloudy_sky.png")

## Add 250 features to original csv
img_data = pd.DataFrame(X)
final = pd.concat([final, img_data],axis=1)
final.drop("Path", axis=1).dropna(axis=1).to_csv('cleaned_data.csv', index=False)

