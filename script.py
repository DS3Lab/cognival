import argparse
from datetime import datetime

#own modules
from handlers.dataHandler import  dataHandler
from handlers.modelHandler import modelHandler
from handlers.fileHandler import *


def handler(config, wordEmbedding, cognitiveData, feature):

    words_test, X_train, y_train, X_test, y_test = dataHandler(config,wordEmbedding,cognitiveData,feature)
    print(len(words_test))
    print(words_test[0].shape)
    print(X_train[0].shape)
    print(y_train[0].shape)
    print(X_test[0].shape)
    print(y_test[0].shape)
    word_error, grids_result, mserrors = modelHandler(config["cogDataConfig"][cognitiveData]["wordEmbSpecifics"][wordEmbedding],
                                         words_test, X_train, y_train, X_test, y_test)

    print("\nAFTER MODEL HANDLER")
    print("word_error shape ")
    print(word_error.shape)
    print("MSERRORS")
    print(mserrors)
    print(type(mserrors))



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

    print("AFTER HANDLER")

    history = {'loss':[],'val_loss':[]}
    loss_list =[]
    val_loss_list =[]

    ##############################################################################
    #   logging results
    ##############################################################################

    for i in range(len(grids_result)):
        fold = {}
        logging['folds'].append(fold)
        # BEST PARAMS APPENDING
        for key in grids_result[i].best_params_:
            logging['folds'][i][key.upper()] = grids_result[i].best_params_[key]
        if config['cogDataConfig'][cognitiveData]['type'] == "multiple_output":
            print("multivariate result")
            # logging['folds'][i]['MSE_PREDICTION:'] = list(mserrors[i])
            # logging['folds'][i]['MSE_PREDICTION_AV_ALL_DIM:'] = np.mean(mserrors[i])
            # #TODO CHECK IF THIS WORKS
            # logging['folds'][i]['LOSS: '] = grids_result[i].best_estimator_.model.history.history['loss']
            # logging['folds'][i]['VALIDATION_LOSS: '] = grids_result[i].best_estimator_.model.history.history['val_loss']
        else:
            print("univariate output")
            logging['folds'][i]['MSE_PREDICTION:'] = mserrors[i]
            print("type of mse_prediction")
            print(type(logging['folds'][i]['MSE_PREDICTION:']))
            logging['folds'][i]['LOSS: '] = grids_result[i].best_estimator_.model.history.history['loss']
            print(type(logging['folds'][i]['LOSS: ']))
            logging['folds'][i]['VALIDATION_LOSS: '] = grids_result[i].best_estimator_.model.history.history['val_loss']
            print(type(logging['folds'][i]['VALIDATION_LOSS: ']))
        #TODO: CHECK WHAT HAPPENS HERE
        loss_list.append(np.array(grids_result[i].best_estimator_.model.history.history['loss'],dtype='float'))
        val_loss_list.append(np.array(grids_result[i].best_estimator_.model.history.history['val_loss'], dtype='float'))

    if config['cogDataConfig'][cognitiveData]['type'] == "multiple_output":
        mserrors = np.array(mserrors, dtype='float')
        mse = np.mean(mserrors, axis=0)
        logging['AVERAGE_MSE'] = list(mse)
        logging['AVERAGE_MSE_AV_ALL_DIM']= np.mean(mse)
    else:
        print("univariate output")
        mse = np.array(mserrors, dtype='float').mean()
        logging['AVERAGE_MSE'] = mse


    ##############################################################################
    #   Prepare results for plot
    ##############################################################################

    #TODO: CHECK PLOTS
    #TODO: CHECK AVERAGE HERE
    #TODO: go over elem in list and average result to create plot
    #TODO: see dimensionality of vectors
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
                        nargs='?', default='config/setupConfig.json')
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