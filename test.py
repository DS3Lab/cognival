from __future__ import division
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import KFold


def dataHandler(wEmb, cogData, feature, dim,config):

    # READ Datasets into dataframes
    df_cD = pd.read_csv(cogData, sep=" ")
    df_wE = pd.read_csv(wEmb, sep=" ",
                         encoding="utf-8", quoting=csv.QUOTE_NONE)

    # with pd.option_context('display.precision', 10):
    #     print(df_cD)

    # print(df_cD)
    if dim=="single":
        df_cD = df_cD[['word',feature]]
    df_cD.dropna(inplace=True)
     
    # Create chunks of df to perform 'MemorySafe'-join
    #df_join = df_cD
    #print(df_join)
    #rows = df_wE.shape[0]
    #chunk_size = rows // 4 
    #rest = rows % 4
    #for i in range(0,4):
    #    begin = chunk_size *i
    #    end = chunk_size * (i+1)
    #    if i==0:
    #        df_join = pd.merge(df_join,df_wE.iloc[begin:end,:],how='left',on = ['word'])
    #    else:
    #        if i == 4-1:
    #            end = end + rest
    #        df_join = pd.merge(df_join,df_wE.iloc[begin:end,:],how='left',on = ['word'])
    #    print(df_join)
        
     
    # # Left (outer) Join to get wordembedding vectors for all words in cognitive dataset
    df_join = pd.merge(df_cD, df_wE, how='left', on=['word'])
    # #df_join = df_cD.join(df_wE,on=['word'],how='left')
    df_join.dropna(inplace=True)
    #
    #print(df_join)
    #
    # words = df_join['word']
    # words = np.array(words, dtype='str').reshape(-1,1)
    #
    # df_join.drop(['word'], axis=1, inplace=True)
    #
    # if dim == "single":
    #     y = df_join[feature]
    #     y = np.array(y, dtype='float').reshape(-1, 1)
    #
    #     X = df_join.drop(feature, axis=1)
    #     X = np.array(X, dtype='float')
    # else:
    #     features = config
    #     y = df_join[features]
    #     y = np.array(y, dtype='float')
    #
    #     X = df_join.drop(features, axis=1)
    #     X = np.array(X, dtype='float')
    #
    # print(words)
    # print(y)
    # print(X)
    print('SUCCESS')

    return 0

def main():
    emb = "/home/delatvan/Dropbox/university/ETH/4fs/projektArbeit/datasets/embeddings/fasttext/crawl-300d-2M.vec"
    #emb = "/home/delatvan/Dropbox/university/ETH/4fs/projektArbeit/datasets/embeddings/word2vec/word2vec.txt"
    #emb = "/home/delatvan/Dropbox/university/ETH/4fs/projektArbeit/datasets/embeddings/fasttext/wiki-news-300d-1M.vec"
    #emb = "/home/delatvan/Dropbox/university/ETH/4fs/projektArbeit/datasets/embeddings/wordnet2vec/wnet2vec_brain.txt"
    all_data = "/home/delatvan/Dropbox/university/ETH/4fs/projektArbeit/datasets/cognitive-data/gaze/all/all_scaled.txt"
    config =["ffd","fpd","tfd", "nfix","mfd","gpt"]
    dataHandler(emb,all_data,"gpt",dim='single', config=config)

if __name__=="__main__":
    main()
