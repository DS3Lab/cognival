import json
import os
import sys
from multiprocessing import Pool
from datetime import  datetime
import script
from handlers.fileHandler import getConfig
from handlers.fileHandler import *
from sideCode.animatedLoading import animatedLoading

#TODO: clear all WARNINGS!


def main(controllerConfig):

    startTime = datetime.now()

    with open(controllerConfig, 'r') as fileReader:
        data = json.load(fileReader)

    config = getConfig(data["configFile"])

    options = []
    #GENERATE all possible case scenarios:
    for cognitiveData in data["cognitiveData"]:
        for feature in data["cognitiveData"][cognitiveData]["features"]:
            for wordEmbedding in data["wordEmbeddings"]:
                option = {"cognitiveData": "empty", "feature": "empty", "wordEmbedding": "empty"}
                option["cognitiveData"] = cognitiveData
                option["feature"] = feature
                option["wordEmbedding"]=wordEmbedding
                options.append(option)

    loggings = []
    word_errors = []
    histories = []

    ##############################################################################
    #   Serialized version
    ##############################################################################

    # for option in options:
    #     print("OPTION\n")
    #     print(option)
    #     logging, word_error, history = script.run(config,option['wordEmbedding'],option['cognitiveData'],option['feature'])
    #     loggings.append(logging)
    #     word_errors.append(word_error)
    #     histories.append(history)


    ##############################################################################
    #   Parallelized version
    ##############################################################################

    pool = Pool(processes=os.cpu_count())
    async_results = [pool.apply_async(script.run,args=(config,
                                           options[i]["wordEmbedding"],
                                           options[i]["cognitiveData"],
                                           options[i]["feature"])) for i in range(len(options))]
    pool.close()

    while (False in [async_results[i].ready() == True for i in range(len(async_results))]):
        completed = [async_results[i].ready() == True for i in range(len(async_results))].count(True)
        animatedLoading(completed, len(async_results))

    pool.join()

    for p in async_results:
        logging, word_error, history = p.get()
        loggings.append(logging)
        word_errors.append(word_error)
        histories.append(history)


    ##############################################################################
    #   Store results
    ##############################################################################

    for i in range(0,len(loggings)):
        writeResults(updateVersion(data["configFile"]),loggings[i],word_errors[i],histories[i])


    timeTaken = datetime.now() - startTime
    print('\n' + str(timeTaken))

    pass



if __name__=="__main__":
    main("config/controllerConfig.json")





