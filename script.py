import argparse
from datetime import datetime

#own modules
from handlers.dataHandler import  dataHandler
from handlers.modelHandler import modelHandler
from handlers.fileHandler import *


def handler(config, wordEmbedding, cognitiveData, feature):

    words_test, X_train, y_train, X_test, y_test = dataHandler(config,wordEmbedding,cognitiveData,feature)
    word_error, grids_result, mserrors = modelHandler(config["cogDataConfig"][cognitiveData]["wordEmbSpecifics"][wordEmbedding],
                                         words_test, X_train, y_train, X_test, y_test)

    return word_error, grids_result, mserrors


def run(config, wordEmbedding, cognitiveData, feature):

    ##############################################################################
    #   Create logging information
    ##############################################################################

    logging = {"folds":[]}

    logging["wordEmbedding"] = wordEmbedding
    logging["cognitiveData"] = cognitiveData
    logging["feature"] = feature

    ##############################################################################
    #   Run model
    ##############################################################################

    startTime = datetime.now()

    word_error, grids_result, mserrors = handler(config, wordEmbedding, cognitiveData, feature)

    history = {'loss':[],'val_loss':[]}
    loss_list =[]
    val_loss_list =[]

    ##############################################################################
    #   logging results
    ##############################################################################

    for i in range(len(grids_result)):
        fold = {}
        logging['folds'].append(fold)
        for key in grids_result[i].best_params_:
            logging['folds'][i][key.upper()] = grids_result[i].best_params_[key]
        logging['folds'][i]['MSE_PREDICTION:'] = mserrors[i]
        logging['folds'][i]['LOSS: '] = grids_result[i].best_estimator_.model.history.history['loss']
        logging['folds'][i]['VALIDATION_LOSS: '] = grids_result[i].best_estimator_.model.history.history['val_loss']
        loss_list.append(np.array(grids_result[i].best_estimator_.model.history.history['loss'],dtype='float'))
        val_loss_list.append(np.array(grids_result[i].best_estimator_.model.history.history['val_loss'], dtype='float'))

    mse = np.array(mserrors, dtype='float').mean()
    logging['AVERAGE_MSE'] = mse


    ##############################################################################
    #   Prepare results for plot
    ##############################################################################

    history['loss'] = np.mean([loss_list[i] for i in range (len(loss_list))],axis=0)
    history['val_loss'] = np.mean([val_loss_list[i] for i in range(len(val_loss_list))], axis=0)

    timeTaken = datetime.now() - startTime
    logging["timeTaken"] = str(timeTaken)

    return logging, word_error, history

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

    config = updateVersion(configFile)

    ##############################################################################
    #   Check for correct data inputs
    ##############################################################################

    while (wordEmbedding not in config['wordEmbConfig']):
        wordEmbedding = input("ERROR Please enter correct wordEmbedding:\n")
        if wordEmbedding == "x":
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

    startTime = datetime.now()

    logging, word_error, history = run(config, wordEmbedding, cognitiveData, feature)

    ##############################################################################
    #   Saving results
    ##############################################################################

    writeResults(config,logging,word_error,history)

    timeTaken = datetime.now() - startTime
    print(timeTaken)

    pass


if __name__ == "__main__":
    main()