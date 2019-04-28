import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import KFold


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
        y = np.array(y, dtype='float').reshape(-1, 1)
    else:
        print("NON EXISTENT CODE, please return later")
        exit(0)

    features = config['cogDataConfig'][cognitiveData]['features']
    X = df_join.drop(features, axis=1)
    X = np.array(X, dtype='float')

    return split_folds(X,y, config["folds"], config["seed"] )

def split_folds(X,y, folds,seed):
    '''

    :param X: np.array
    :param y: np.array
    :param folds: number of folds
    :return: X_train = [trainingset1, trainingset2, trainingset3,...]
    '''

    np.random.seed(seed)
    np.random.shuffle(X)

    np.random.seed(seed)
    np.random.shuffle(y)


    kf = KFold(n_splits=folds, shuffle=False, random_state=None)
    kf.get_n_splits(X)

    X_train = []
    y_train = []

    X_test =[]
    y_test =[]

    for train_index, test_index in kf.split(X):
        X_train.append(X[train_index])
        y_train.append(y[train_index])
        X_test.append(X[test_index])
        y_test.append(y[test_index])

    return X_train, y_train, X_test, y_test
