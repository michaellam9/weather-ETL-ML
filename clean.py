import pandas as pd
import glob
from scipy import ndimage
import os.path
import numpy as np

def extractDate(filename):
    name = os.path.basename(filename)
    name = name.split('-')[1]
    date = name[:4] + '-' + name[4:6] + '-' + name[6:8]
    time = name[8:10] + ':' + name[10:12]
    return date + ' ' + time

l = [pd.read_csv(filename, skiprows=16) for filename in glob.glob('yvr-weather/*.csv')]
data = pd.concat(l, axis=0).dropna(subset = ['Weather'])

images = [(extractDate(filename), filename) for filename in glob.glob('katkam-scaled/*.jpg')]

img_data = pd.DataFrame(images, columns=['Date/Time', 'Path'])

final = data.merge(img_data, on=['Date/Time'])
final.to_csv('cleaned_data.csv', index=False)
print(final)
