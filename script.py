from multiprocessing.pool import ThreadPool
import logging
from datetime import datetime

#own modules
from animatedLoading import  animatedLoading
from dataHandler import  dataHandler
from modelHandler import modelHandler
from plotHandler import plotHanlder
from eyetrackingHandler import eyetrackingHandler

#TODO: don't hardcode all inputs, create command line interface
#cognitiveData = "../datasets/eeg/zuco/zuco_scaled.txt"
cognitiveData ='../datasets/dundee/dundee_scaled.txt'
#TODO: SET to choose for specific dimension
wordEmbDir = "../datasets/glove-6B/glove.6B.50d.copy.txt"

def run():

    #X_train, y_train = dataHandler(cognitiveData,wordEmbDir)
    X_train, y_train = eyetrackingHandler(cognitiveData, wordEmbDir)
    history = modelHandler(X_train, y_train)

    return history


def main():
    epochs = 400

    logging.basicConfig(filename='history.log', level=logging.DEBUG,
                        format='%(asctime)s:%(message)s')

    # pool = ThreadPool(processes=1)

    startTime = datetime.now()
    # async_result = pool.apply_async(run)
    #
    # while(async_result.ready()==False):
    #     animatedLoading()

    history = run()

    timeTaken = datetime.now()-startTime
    print(timeTaken)
    logging.info(timeTaken)

    # history = async_result.get()
    plotHanlder(history, startTime)

    logging.info('EPOCHS')
    logging.info(epochs)
    logging.info('LOSS')
    logging.info(history.history['loss'])
    logging.info('VALIDATION LOSS')
    logging.info(history.history['val_loss'])

    pass


if __name__ == "__main__":
    main()