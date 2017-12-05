Webcams, Predictions, and Weather 
==============

For CMPT 318
--------------



**Required Libraries** 
 - pandas
 - matplotlib
 - numpy
 - sklearn
 - scipy
 - statsmodels
 - PIL






1. Clean.py removes all rows with no weather description. All columns with mostly NaN are also removed. The dates and times are converted to a pandas Datetime object. Images are read into memory and a PCA is done to reduce the features of the images. The final csv is the image data reduced to 250 features with each feature a seperate column. Each image is joined with the other features based on the time the picture was taken and the data was recorded.

2. Analysis.py runs a comparison between different machine learning models with and without image data. "python3 analysis.py" is the command to run the analysis with a table of the results printed.
