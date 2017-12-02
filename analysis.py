import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def to_timestamp(date):
    return date.timestamp()

def compareModels(X_train, X_test, y_train, y_test, models):
    scores = []
    times = []
    for i, m in enumerate(models):
        t0 = time.time()
        m.fit(X_train, y_train)
        t1 = time.time()
        score = m.score(X_test, y_test)
        time_elapsed = t1-t0
        times.append(time_elapsed)
        scores.append(score)
    
    stats = pd.DataFrame(data={'Score':scores, 'Model':['SVC','GaussianNB','KNN'],'Time':times})
    return stats


def main():

    data = pd.read_csv('cleaned_data.csv', parse_dates=['Date/Time'])
    data['timestamp'] = data['Date/Time'].apply(to_timestamp)
    random = 15
    y = data['Weather']

    model_SVC = make_pipeline(
        StandardScaler(),
        SVC(C=10))

    model_Gaussian = make_pipeline(
        StandardScaler(),
        GaussianNB())

    model_Knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier())


    models = [model_SVC, model_Gaussian, model_Knn]

    allScores =[]
    
    ## Run multiple trials with different splits to get a more accurate score
    for i in range(5):
    
        cols = ['Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)', 'Stn Press (kPa)','timestamp']
        X = data[cols]
        X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=i)
        statsDataOnly = compareModels(X_train,X_test,y_train,y_test,models)
        statsDataOnly['Model'] = statsDataOnly['Model'] + ' (Data only)'

        X = data.loc[:,'0':'249']
        X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=i)
        statsImagesOnly = compareModels(X_train,X_test,y_train,y_test,models)
        statsImagesOnly['Model'] = statsImagesOnly['Model'] + ' (Images only)'

        X = data.drop(['Weather','Date/Time'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=i)
        statsBoth = compareModels(X_train,X_test,y_train,y_test,models)
        statsBoth['Model'] = statsBoth['Model'] + ' (Data and Images)'
    

        trial = pd.concat([statsDataOnly,statsImagesOnly, statsBoth])
        allScores.append(trial)


    final_all = pd.concat(allScores).groupby('Model').mean().reset_index()
    Gauss_scores = final_all[ final_all['Model'].str.contains('Gaussian') ]
    SVC_scores = final_all[ final_all['Model'].str.contains('SVC') ]
    KNN_scores = final_all[ final_all['Model'].str.contains('KNN') ]

    ## Plot multiple bar charts with scores
    X = np.arange(3)
    plt.bar(X + 0.00, Gauss_scores['Score'], color = 'b', width = 0.25, label='Gaussian')
    plt.bar(X + 0.25, SVC_scores['Score'], color = 'g', width = 0.25, label='SVC')
    plt.bar(X + 0.50, KNN_scores['Score'], color = 'r', width = 0.25, label='KNN')
    plt.legend(['Data and Images', 'Data Only', 'Images Only'])
    plt.xticks(X+0.25,('Gaussian','SVC','KNN'))
    plt.ylim(0,1)
    plt.ylabel('Score')
    plt.title('Scores of different models with subset of features')
    plt.show()

if __name__ == '__main__':
    main()
