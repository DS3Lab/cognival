import json
import os
from multiprocessing import Pool, Lock
from datetime import  datetime
import script
from handlers.fileHandler import getConfig
from handlers.fileHandler import *
from sideCode.animatedLoading import animatedLoading



def main(controllerConfig):

    #TODO: SOLVE PROBLEM WITH LOGGING
    #TODO: logging, writing to file and ploting not safe inside controller nor parallelize
    #TODO: logging, writing to file and plotting outside of main

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

    for option in options:
        logging, word_error, history = script.run(config,option['wordEmbedding'],option['cognitiveData'],option['feature'])
        loggings.append(logging)
        word_errors.append(word_error)
        histories.append(history)


    ##############################################################################
    #   Parallelized version
    ##############################################################################

    # pool = Pool(processes=os.cpu_count())
    # async_results = [pool.apply_async(script.run,args=(data["configFile"],
    #                                        options[i]["wordEmbedding"],
    #                                        options[i]["cognitiveData"],
    #                                        options[i]["feature"])) for i in range(len(options))]
    # pool.close()
    # pool.join()


    # while(async_results[len(async_results)-1].ready()!=True):
    #     lock.acquire()
    #     animatedLoading()
    #     lock.release()

    # results = [p.get() for p in async_results]

    # for result in results:
    #     plotHandler(result['title'], result['history'], result['version'], result['outputDir'])

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





