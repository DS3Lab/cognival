import numpy as np
import pandas as pd
from numpy import fromfile, dtype
from gensim.models.keyedvectors import KeyedVectors

def bin_to_df(filename,rec_dtype):
    return pd.DataFrame(fromfile(filename,rec_dtype))

def bin_to_txt(binPath,binName,outputName):
    model = KeyedVectors.load_word2vec_format(binPath+binName,binary=True)
    model.save_word2vec_format(binPath+outputName,binary=False)

# def main(binPath,binName,outputName, dim):
#     header = [('word','str')] + [('x%s'%i,'float64') for i in range(1,dim+1)]
#     dt = dtype(header)
#     print(bin_to_df(binPath+binName,dt))
#     #bin_to_txt(binPath,binName,outputName)

# if __name__=="__main__":
#     main('/Users/delatan/Dropbox/university/ETH/4fs/projektArbeit/datasets/embeddings/word2vec/',
#                   "GoogleNews-vectors-negative300.bin", 'word2vec.txt',300)

