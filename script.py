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


def handler(config, wordEmbedding, cognitiveData, feature):

    words_test, X_train, y_train, X_test, y_test = dataHandler(config,wordEmbedding,cognitiveData,feature)
    word_error, grids_result, mserrors = modelHandler(config["cogDataConfig"][cognitiveData]["wordEmbSpecifics"][wordEmbedding],
                                         words_test, X_train, y_train, X_test, y_test)

    return word_error, grids_result, mserrors


def run(configFile, wordEmbedding, cognitiveData, feature):

    config = updateVersion(configFile)

    ##############################################################################
    #   Check for correct data inputs
    ##############################################################################

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

    #TODO: pass this to configuration file
    outputDir = config['outputDir']
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    logging.basicConfig(filename=outputDir+"/"+str(+config['version'])+'.log', level=logging.DEBUG,
                        format='%(asctime)s:%(message)s')
    logging.info("Word Embedding: "+wordEmbedding)
    logging.info("Cognitive Data: " + cognitiveData)
    logging.info("Feature: "+feature)

    ##############################################################################
    #   Run model
    ##############################################################################

    startTime = datetime.now()

    word_error, grids_result, mserrors = handler(config, wordEmbedding, cognitiveData, feature)

    np.savetxt(outputDir+"/"+str(+config['version'])+'.txt', word_error, delimiter=" ", fmt="%s")

    history = {'loss':[],'val_loss':[]}
    loss_list =[]
    val_loss_list =[]

    ##############################################################################
    #   logging results
    ##############################################################################

    for i in range(len(grids_result)):
        logging.info(" FOLD: "+str(i))
        for key in grids_result[i].best_params_:
            logging.info(key.upper())
            logging.info(grids_result[i].best_params_[key])
        logging.info('MSE PREDICTION:')
        logging.info(mserrors[i])
        logging.info('LOSS: ')
        logging.info(grids_result[i].best_estimator_.model.history.history['loss'])
        logging.info('VALIDATION LOSS: ')
        logging.info(grids_result[i].best_estimator_.model.history.history['val_loss'])
        loss_list.append(np.array(grids_result[i].best_estimator_.model.history.history['loss'],dtype='float'))
        val_loss_list.append(np.array(grids_result[i].best_estimator_.model.history.history['val_loss'], dtype='float'))

    logging.info('AVERAGE MSE from all folds')
    mse = np.array(mserrors, dtype='float').mean()
    logging.info(mse)

    ##############################################################################
    #   Prepare results for plot
    ##############################################################################

    history['loss'] = np.mean([loss_list[i] for i in range (len(loss_list))],axis=0)
    history['val_loss'] = np.mean([val_loss_list[i] for i in range(len(val_loss_list))], axis=0)

    title = wordEmbedding+' '+cognitiveData+' '+feature
    plotHandler(title,history,config['version'],outputDir)

    timeTaken = datetime.now() - startTime
    print('\n' + str(timeTaken))
    logging.info(" TIME TAKEN:" + str(timeTaken))
    logging.shutdown()

    ##############################################################################
    #   Parallelized version
    ##############################################################################

    # result ={}
    # result["title"] = title
    # result["history"] = history
    # result["version"] = config["version"]
    # result["outputDir"] = outputDir

    # return result
    return timeTaken

def main():

    ##############################################################################
    #   Set up of command line arguments to run the script
    ##############################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument("configFile", help="path and name of configuration file",
                        nargs='?', default='config/c.json')
    parser.add_argument("-c", "--cognitiveData", type=str, default=None,
                        help="cognitiveData to train the model")
    parser.add_argument("-f", "--feature", type=str,
                        default=None, help="feature of the dataset to train the model")
    parser.add_argument("-w", "--wordEmbedding", type=str, default=None,
                        help="wordEmbedding to train the model")

    args = parser.parse_args()

    configFile = args.configFile
    cognitiveData = args.cognitiveData
    feature = args.feature
    wordEmbedding = args.wordEmbedding

    run(configFile, wordEmbedding, cognitiveData, feature)

    ##############################################################################
    #   Parallelized version
    ##############################################################################

    # result = run(configFile, wordEmbedding, cognitiveData, feature)
    # plotHandler(result['title'], result['history'], result['version'], result['outputDir'])

    pass


if __name__ == "__main__":
    main()