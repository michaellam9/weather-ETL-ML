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

**Order of Execution**
1. $ python3 clean.py
  * The program expects the "katkam-scaled" and "yvr-weather" folders to be in the same directory. 
  * Clean.py removes all rows with no weather description. All columns with mostly NaN are also removed. The dates and times are converted to a pandas Datetime object. Images are read into memory and a PCA is done to reduce the features of the images. The final csv is the image data reduced to 250 features with each feature a seperate column. Each image is joined with the other features based on the time the picture was taken and the time the data was recorded.
  * Produces cleaned_data.csv, which is required to run all further programs. Also produces four sample images showing how the PCA transformation affects the image data. 
  * Note: A virtual machine with 6GB of RAM ran out of memory when running clean.py (due to the PCA transformation), so it is recommended to run this on the CSIL computers. 

2. $ python3 get_unique_weather.py
  * This program can be optionally run to find out more about how the short descriptions for the weather are written, and how frequently each weather description occurs. This program does not produce anything but it aided us in figuring out what to do with unclear weather descriptions ("Rain" vs. "Rain Showers"), as well as weather descriptions with only a few occurrences. 

3. $ python3 analysis.py
  * This program compares three machine learning models (GaussianNB, K-Nearest Neighbors, Support Vector Machine) with three different sets of features (image features only, weather instrument features only, both sets of features). 
  * Produces results.png, which is a bar plot showing the results of the nine different cases. 

4. $ python3 SVC_ANOVA_analysis.py
  * This program runs 40 trials each on the three feature sets, but only with the SVM model. An ANOVA test is run on the data, followed by a post hoc Tukey test. 
  * Produces posthoc_tukey_plot.png, which is a plot of the three feature sets and the range of their accuracy scores, as produced by the Tukey test. 
