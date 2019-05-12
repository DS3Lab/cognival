import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import KFold


def dataHandler(wEmb, cogData, feature):
    #TODO: write code for Multidimensional

    # READ Datasets into dataframes
    df_cD = pd.read_csv(cogData, sep=" ")
    df_wE = pd.read_csv(wEmb, sep=" ",
                        encoding="utf-8", quoting=csv.QUOTE_NONE)

    # Left (outer) Join to get wordembedding vectors for all words in cognitive dataset
    df_join = pd.merge(df_cD, df_wE, how='left', on=['word'])
    df_join.dropna(inplace=True)

    print(df_join)

    # words = df_join['word']
    # words = np.array(words, dtype='str').reshape(-1,1)
    #
    # df_join.drop(['word'], axis=1, inplace=True)
    #
    # if config['cogDataConfig'][cognitiveData]['type'] == "single_output":
    #     y = df_join[feature]
    #     y = np.array(y, dtype='float').reshape(-1, 1)
    # else:
    #     print("NON EXISTENT CODE, please return later")
    #     exit(0)
    #
    # features = config['cogDataConfig'][cognitiveData]['features']
    # X = df_join.drop(features, axis=1)
    # X = np.array(X, dtype='float')

    return 0

def main():

    dataHandler("/Users/delatan/Dropbox/university/ETH/4fs/projektArbeit/datasets/embeddings/fasttext/crawl-300d-2M.vec",
                "/Users/delatan/Dropbox/university/ETH/4fs/projektArbeit/datasets/cognitive-data/gaze/all/all_scaled.txt","ffd")

if __name__=="__main__":
    main()