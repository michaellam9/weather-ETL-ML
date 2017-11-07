import pandas as pd
import glob

l = [pd.read_csv(filename, skiprows=16) for filename in glob.glob('yvr-weather/*.csv')]
data = pd.concat(l, axis=0).dropna(subset = ['Weather']).reset_index()
print(data)
print(data['Weather'].unique())
