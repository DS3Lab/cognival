import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  #disable tensorflow debugging
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

    ##############################################################################
    #   OPTION GENERATION
    ##############################################################################

    print("\nGENERATING OPTIONS...")

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

    print("\nSUCCESSFUL OPTIONS GENERATION")

    ##############################################################################
    #   JOINED DATAFRAMES GENERATION
    ##############################################################################




    ##############################################################################
    #   Parallelized version
    ##############################################################################

    print("\nMODELS CREATION, FITTING, PREDICTION...\n ")

    # proc = os.cpu_count()
    proc = 20
    pool = Pool(processes=proc)
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

    print("\nSUCCESSFUL MODELS")

    ##############################################################################
    #   Store results
    ##############################################################################

    print("\nSTORING RESULTS...")

    all_runs = {}

    for i in range(0,len(loggings)):
        config = getConfig(data["configFile"])
        writeResults(config, loggings[i], word_errors[i], histories[i])
        options[i]["AVERAGE_MSE"] = loggings[i]["AVERAGE_MSE"]
        all_runs[config["version"]] = options[i]
        updateVersion(data["configFile"])

    writeOptions(config,all_runs)

    print("\nSUCCESSFUL STORING")

    print("\nSUCCESSFUL RUN")

    timeTaken = datetime.now() - startTime
    print('\n' + str(timeTaken))



    pass


if __name__=="__main__":
    #main("config/controllerConfig.json")
    main("config/c.json")




