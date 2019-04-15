from multiprocessing.pool import ThreadPool
from animatedLoading import  animatedLoading
from datetime import datetime
import time
from dataHandler import  dataHandler

#TODO: don't hardcode all inputs, create command line interface
cognitiveData = "../datasets/eeg/zuco/zuco_scaled.txt"
#TODO: SET to choose for specific dimension
wordEmbDir = "../datasets/glove-6B/glove.6B.50d.copy.txt"

def run():
    X_train, y_train, X_test, y_test = dataHandler(cognitiveData,wordEmbDir)
    return 0


def main():

    pool = ThreadPool(processes=1)

    startTime = datetime.now()
    async_result = pool.apply_async(run)

    while(async_result.ready()==False):
        animatedLoading()

    print(datetime.now()-startTime)

    return_val =async_result.get()

    #print(return_val)

    pass


if __name__ == "__main__":
    main()