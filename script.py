import os
import logging
import argparse
import numpy as np
from datetime import datetime

#own modules
from handlers.dataHandler import  dataHandler
from handlers.modelHandler import modelHandler
from handlers.plotHandler import plotHandler
from handlers.fileHandler import *


def run(config,wordEmbedding,cognitiveData,feature):

    words_test, X_train, y_train, X_test, y_test = dataHandler(config,wordEmbedding,cognitiveData,feature)
    word_error, grids_result, mserrors = modelHandler(config["cogDataConfig"][cognitiveData]["wordEmbSpecifics"][wordEmbedding],
                                         words_test, X_train, y_train, X_test, y_test)

    return word_error, grids_result, mserrors


def main():

    ##############################################################################
    #   Set up of command line arguments to run the script
    ##############################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument("configFile",help="path and name of configuration file",
                        nargs='?', default='config/config.json')
    parser.add_argument("-d","--dataset", type=str,default=None,
                        help="dataset to train the model")
    parser.add_argument("-f","--feature", type=str,
                        default=None, help="feature of the dataset to train the model")
    parser.add_argument("-w","--wordEmbedding", type=str, default=None,
                        help="wordEmbedding to train the model")

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

    logging.basicConfig(filename=outputDir+"/"+str(+config['version'])+'.log', level=logging.DEBUG,
                        format='%(asctime)s:%(message)s')
    logging.info("Word Embedding: "+wordEmbedding)
    logging.info("Cognitive Data: " + cognitiveData)
    logging.info("Feature: "+feature)


    startTime = datetime.now()

    word_error, grids_result, mserrors = run(config, wordEmbedding, cognitiveData, feature)

    np.savetxt(outputDir+"/"+str(+config['version'])+'.txt', word_error)

    timeTaken = datetime.now()-startTime
    print('\n'+str(timeTaken))
    logging.info(" TIME TAKEN:"+str(timeTaken))

    #TODO: create numpy array from history loss and val_loss, store all 5 into list. and this one inside dictionary history
    # pass this to plotHandler to plot average of results, calculate mean before, and finally change title of plot to cogdata,feature,wordemb
    # return grid_result.best_estimator_.model.history
    # plotHandler(history, config['version'],outputDir)

    for i in range(len(grids_result)):
        logging.info("\n",i," FOLD: ")
        for key in grids_result[i].best_params_:
            logging.info(key.upper())
            logging.info(grids_result[i].best_params_[key])
        logging.info('MSE PREDICTION:')
        logging.info(mserrors[i])
        logging.info('LOSS: ')
        logging.info(grids_result[i].best_estimator_.model.history.history['loss'])
        logging.info('VALIDATION LOSS: ')
        logging.info(grids_result[i].best_estimator_.model.history.history['val_loss'])

    logging.info('AVERAGE MSE from all folds')
    mse = np.array(mserrors,dtype='float').mean()
    logging.info(mse)
    pass


if __name__ == "__main__":
    main()