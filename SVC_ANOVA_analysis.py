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

from scipy import stats

def to_timestamp(date):
    return date.timestamp()

model_SVC = make_pipeline(
    StandardScaler(),
    SVC(C=10)
)
def run_40_tests(X,y,name):
    scores = []
    for i in range(40):
        X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=i)
        model_SVC.fit(X_train,y_train)
        score = model_SVC.score(X_test, y_test)
        scores.append(score)
    return pd.Series(scores, name=name)

def main():

    data = pd.read_csv('cleaned_data.csv', parse_dates=['Date/Time'])
    data['timestamp'] = data['Date/Time'].apply(to_timestamp)

    y = data['Weather']
    
    cols = ['Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)', 'Stn Press (kPa)','timestamp']
    X = data[cols]
    instruments_data_only = run_40_tests(X,y,"Instrument Data")
    # instruments_data_only.hist()
    # plt.show()

    X = data.loc[:,'0':'249']
    picture_data_only = run_40_tests(X,y,"Photo Data")
    # picture_data_only.hist()
    # plt.show()

    X = data.drop(['Weather','Date/Time'], axis=1)
    instrument_and_picture_data = run_40_tests(X,y,"Both Data Sets")
    # instrument_and_picture_data.hist()
    # plt.show()

    anova = stats.f_oneway(instruments_data_only, picture_data_only, instrument_and_picture_data)
    print("ANOVA p-value:", anova.pvalue)

    melt_data = pd.melt(pd.concat([instruments_data_only, picture_data_only, instrument_and_picture_data], axis=1))

    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    posthoc = pairwise_tukeyhsd(melt_data['value'], melt_data['variable'],alpha=0.05)
    print(posthoc)

    fig = posthoc.plot_simultaneous()
    plt.savefig("posthoc_tukey_plot.png")



    ## Plot multiple bar charts with scores
    # X = np.arange(3)
    # plt.bar(0, SVC_scores['Score'], color = 'g', width = 0.25, label='SVC')
    # plt.legend(['Data and Images', 'Data Only', 'Images Only'])
    # plt.xticks(0,('SVC'))
    # plt.ylim(0,1)
    # plt.ylabel('Score')
    # plt.title('Scores of different models with subset of features')
    # plt.show()

if __name__ == '__main__':
    main()
