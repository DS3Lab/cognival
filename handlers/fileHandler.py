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

    title = logging["cognitiveData"] + '_' + logging["feature"] + '_' + logging["wordEmbedding"]+'_'+str(config["version"])

    outputDir = config['outputDir']+"/"+title
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    with open(outputDir+"/"+title+'.json','w') as fileWriter:
        json.dump(logging,fileWriter,indent=4, sort_keys=True)

    np.savetxt(outputDir + "/" + title + '.txt', word_error, delimiter=" ", fmt="%s")

    plotHandler(title,history,outputDir)

    pass

def writeOptions(config, all_runs):

    outputDir = config['outputDir']

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    with open(outputDir+"/options"+str(config["version"])+'.json','w') as fileWriter:
        json.dump(all_runs,fileWriter, indent=4,sort_keys=True)

    pass
