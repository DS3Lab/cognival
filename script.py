import os
import logging
import argparse
from datetime import datetime

#own modules
from handlers.dataHandler import  dataHandler
from handlers.modelHandler import modelHandler
from handlers.plotHandler import plotHanlder
from handlers.fileHandler import *


def run(config,wordEmbedding,cognitiveData,feature):

    X_train, y_train = dataHandler(config,wordEmbedding,cognitiveData,feature)
    history = modelHandler(X_train, y_train)

    return history


def main():

    ##############################################################################
    #   Set up of command line arguments to run the script
    ##############################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument("configFile",help="path and name of configuration file",
                        nargs='?', default='config/config.json')
    #TODO: change dundee default for later use
    parser.add_argument("-d","--dataset", type=str, choices=["dundee"],
                        default="dundee", help="dataset to train the model")
    parser.add_argument("-f","--feature", type=str, default=None,
                        help="feature of the dataset to train the model")
    #TODO: change wordEmbedding deafualt for future use
    parser.add_argument("-w","--wordEmbedding", type="str", choices=["glove"],
                        default="glove", help="wordEmbedding to train the model")

    args = parser.parse_args()

    configFile = args.configFile
    cognitiveData = args.dataset
    feature = args.feature
    wordEmbedding = args.wordEmbedding

    config = updateVersion(configFile)

    while(wordEmbedding not in config['wordEmbConfig'] ):
        wordEmbedding = input("ERROR Please enter correct wordEmbedding:\n")
        if wordEmbedding=="x":
            exit(0)

    while (cognitiveData not in config['cogDataConfig']):
        cognitiveData = input("ERROR Please enter correct cognitive dataset:\n")
        if cognitiveData == "x":
            exit(0)

    if config['cogDataConfig'][cognitiveData]['type'] == "single_output":
        while feature not in config['cogDataConfig'][cognitiveData]['features']:
            feature = input("ERROR Please enter correct feature for specified cognitive dataset:\n")
            if feature == "x":
                exit(0)

    ##############################################################################
    #   Create logging information and run main program
    ##############################################################################

    outputDir = 'output'
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    logging.basicConfig(filename=outputDir+config['version']+'.log', level=logging.DEBUG,
                        format='%(asctime)s:%(message)s')
    logging.info("Word Embedding: "+wordEmbedding)
    logging.info("Cognitive Data: " + cognitiveData)
    logging.info("Feature: "+feature)

    # pool = ThreadPool(processes=1)

    startTime = datetime.now()
    # async_result = pool.apply_async(run)
    #
    # while(async_result.ready()==False):
    #     animatedLoading()

    history = run(config,wordEmbedding,cognitiveData,feature)

    timeTaken = datetime.now()-startTime
    print('\n'+timeTaken)
    logging.info(timeTaken)

    # history = async_result.get()
    plotHanlder(history, config['version'],outputDir)
    #TODO: print configuration of model fit
    #logging.info('EPOCHS')
    #logging.info(epochs)
    logging.info('LOSS')
    logging.info(history.history['loss'])
    logging.info('VALIDATION LOSS')
    logging.info(history.history['val_loss'])

    pass


if __name__ == "__main__":
    main()