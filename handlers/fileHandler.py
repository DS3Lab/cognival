import json
import os
import numpy as np
from handlers.plotHandler import plotHandler

def updateVersion(configFile):

    with open(configFile, 'r') as fileReader:
        config = json.load(fileReader)

    config['version'] = config['version'] +1

    with open(configFile,'w') as fileWriter:
        json.dump(config,fileWriter, indent=4, sort_keys=True)

    return config

def getConfig(configFile):

    with open(configFile, 'r') as fileReader:
        config = json.load(fileReader)

    return config

def writeResults(config, logging, word_error, history):

    if not os.path.exists(config['outputDir']):
        os.mkdir(config['outputDir'])

    outputDir = config['outputDir']+"/"+str(config['version'])
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    with open(outputDir+"/"+str(config['version'])+'.json','w') as fileWriter:
        json.dump(logging,fileWriter,indent=4, sort_keys=True)

    np.savetxt(outputDir + "/" + str(+config['version']) + '.txt', word_error, delimiter=" ", fmt="%s")

    title = logging["wordEmbedding"]+' '+logging["cognitiveData"]+' '+logging["feature"]
    plotHandler(title,history,config['version'],outputDir)

    pass

def writeOptions(config, options):

    outputDir = config['outputDir']

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)


    with open(outputDir+"/options"+'.txt','w') as fileWriter:
        for item in options:
            fileWriter.write("%s\r\n" %item)

    pass