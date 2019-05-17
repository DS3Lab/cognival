import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import KFold


def update(df1, df2, on_column, columns_to_omit):
    # Both dataframes have to have same column names
    header = list(df1)
    header = header[columns_to_omit:]

    start = df1.shape[1]
    to_update = df1.merge(df2, on=on_column, how='left').iloc[:, start:].dropna()
    to_update.columns = header

    # UPDATE just on NaN values
    # for elem in header:
    # 	df1.loc[df1[elem].isnull(),elem] = to_update[elem]
    # 	print(df1)

    # UPDATE whole row when NaN appears
    df1.loc[df1[header[0]].isnull(), header] = to_update
    return df1


def dataHandler(config, wordEmbedding, cognitiveData, feature):

    # READ Datasets into dataframes
    df_cD = pd.read_csv(config['PATH'] + config['cogDataConfig'][cognitiveData]['dataset'], sep=" ")
    df_wE = pd.read_csv(config['PATH'] + config['wordEmbConfig'][wordEmbedding], sep=" ",
                        encoding="utf-8", quoting=csv.QUOTE_NONE)

    # In case it's a single output cogData we just need the single feature
    if config['cogDataConfig'][cognitiveData]['type'] == "single_output":
        df_cD = df_cD[['word',feature]]
    df_cD.dropna(inplace=True)

    # # Create chunks of df to perform 'MemorySafe'-join
    # chunk_number = 4
    # df_join = df_cD
    # rows = df_wE.shape[0]
    # chunk_size = rows // chunk_number
    # rest = rows % chunk_number
    # for i in range(0, chunk_number):
    #     begin = chunk_size * i
    #     end = chunk_size * (i + 1)
    #     if i == 0:
    #         df_join = pd.merge(df_join, df_wE.iloc[begin:end, :], how='left', on=['word'])
    #     else:
    #         if i == chunk_number - 1:
    #             end = end + rest
    #         update(df_join, df_wE.iloc[begin:end, :], on_column=['word'], columns_to_omit=df_cD.shape[1])

    # Left (outer) Join to get wordembedding vectors for all words in cognitive dataset
    df_join = pd.merge(df_cD, df_wE, how='left', on=['word'])

    df_join.dropna(inplace=True)

    words = df_join['word']
    words = np.array(words, dtype='str').reshape(-1,1)

    df_join.drop(['word'], axis=1, inplace=True)

    if config['cogDataConfig'][cognitiveData]['type'] == "single_output":
        y = df_join[feature]
        y = np.array(y, dtype='float').reshape(-1, 1)

        X = df_join.drop(feature, axis=1)
        X = np.array(X, dtype='float')
    else:
        features = config['cogDataConfig'][cognitiveData]['features']
        y = df_join[features]
        y = np.array(y, dtype='float')

        X = df_join.drop(features, axis=1)
        X = np.array(X, dtype='float')

    return split_folds(words ,X,y, config["folds"], config["seed"] )

def split_folds(words, X, y, folds, seed):
    '''

    :param words: np.array with words, same order, corresponding to X and y vectors
    :param X: np.array
    :param y: np.array
    :param folds: number of folds
    :return: X_train = [trainingset1, trainingset2, trainingset3,...]
    '''

    np.random.seed(seed)
    np.random.shuffle(words)

    np.random.seed(seed)
    np.random.shuffle(X)

    np.random.seed(seed)
    np.random.shuffle(y)


    kf = KFold(n_splits=folds, shuffle=False, random_state=None)
    kf.get_n_splits(X)

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    words_test = []

    for train_index, test_index in kf.split(X):
        X_train.append(X[train_index])
        y_train.append(y[train_index])
        X_test.append(X[test_index])
        y_test.append(y[test_index])
        words_test.append(words[test_index])

    return words_test, X_train, y_train, X_test, y_test
