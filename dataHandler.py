import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



def dataHandler(cognitiveData, wordEmbDir):

    #TODO: Modularize code into segments, read data, join set training/test % and create numpy vectors

    df_cD = pd.read_csv(cognitiveData, sep=" ")

    # engine='python' FIX pandas.errors.ParserError: Error tokenizing data. C error: EOF inside string starting at row 8
    df_wE = pd.read_csv(wordEmbDir, sep=" ",encoding="utf-8", quoting=csv.QUOTE_NONE)

    #Left (outer) Join to get wordembedding vectors for all words in eeg data
    df_join = pd.merge(df_cD,df_wE,how='left', on=['word'])
    df_join = df_join.dropna(inplace=False)
    df_join = df_join.drop(['word'],axis=1)

    #TODO: just reproduce one variable from eeg
    #check for later use cases to do all of them
    cols = [i for i in range(1,105)]
    df_join.drop(df_join.columns[cols],axis=1,inplace=True)


    y =  df_join[df_join.columns[0]]
    X = df_join[df_join.columns[1:]]

    y = np.array(y,dtype='float')
    X = np.array(X,dtype='float')

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1,1))


    # # #TODO: set calculated percentage and not hardcoded number
    #X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2)
    X_train = X_scaled
    y_train = y_scaled


    return X_train, y_train

def datasetJOIN():

    pass