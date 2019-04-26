import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def dataHandler(config, wordEmbedding, cognitiveData, feature):
    #TODO: write code for Multidimensional

    # READ Datasets into dataframes
    df_cD = pd.read_csv(config['PATH'] + config['cogDataConfig'][cognitiveData]['dataset'], sep=" ")
    df_wE = pd.read_csv(config['PATH'] + config['wordEmbConfig'][wordEmbedding], sep=" ",
                        encoding="utf-8", quoting=csv.QUOTE_NONE)

    # Left (outer) Join to get wordembedding vectors for all words in cognitive dataset
    df_join = pd.merge(df_cD, df_wE, how='left', on=['word'])
    df_join.dropna(inplace=True)
    #TODO: NO WORD DROPPING
    #TODO: dictionary word:vec
    df_join.drop(['word'], axis=1, inplace=True)

    if config['cogDataConfig'][cognitiveData]['type'] == "single_output":
        y = df_join[feature]
        y_train = np.array(y, dtype='float').reshape(-1, 1)
    else:
        print("NON EXISTENT CODE, please return later")
        exit(0)

    features = config['cogDataConfig'][cognitiveData]['features']
    X = df_join.drop(features, axis=1)
    X_train = np.array(X, dtype='float')

    # # # #TODO: add validations,prediction and training data
    # #X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2)
    #X_train,X_test,X_validate, y_train, y_test, y_validate


    return X_train, y_train
