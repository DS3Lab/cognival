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
