import numpy as np
import pandas as pd
import json


def createTable(CONFIG):
    with open(CONFIG,'r') as fR:
        config = json.load(fR)
    header = [key for key in config['wordEmbConfig']]
    print(header)
    index1 = []
    index2 = []
    for cD in config["cogDataConfig"]:
    	for feature in config["cogDataConfig"][cD]["features"]:
    		index1.append(cD)
    		index2.append(feature)
    index = [index1,index2]	
    
    setup = {header[j]:[np.NaN for i in range(len(index2)) ] for j in range(len(header))} 
    df = pd.DataFrame(setup,index)
    print(df)	
    
    pass

def fillTable(PATH, table):

    pass

def main():
    OPTIONS = "../test_final/options.json"
    CONFIG = "../config/setupConfig.json"
    createTable(CONFIG)
    pass

if __name__=="__main__":
    main()
