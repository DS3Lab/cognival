import json

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

print(updateVersion('config/config.json'))