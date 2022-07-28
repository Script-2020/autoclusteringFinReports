import pickle
import os
import json
import pandas as pd
from pathlib import Path

class FileManager:

   
    @staticmethod
    def saveDictToExcel(fileName, dictData):
        if not "xls" in fileName:
            fileName += ".xlsx"
        df = pd.DataFrame(data=dictData)
        #df = (df.T)
        df.to_excel(fileName)

    @staticmethod
    def saveJSON(fileName, data):
        if 'json' not in fileName:
            fileName += ".json"
        with open(fileName, 'w') as outfile:
            json.dump(data, outfile)

    @staticmethod
    def readJSON(fileName):
        print('Reading ' , fileName)

        if 'json' not in fileName:
            fileName += ".json"

        if not os.path.exists(fileName):
            return None

        f = open(fileName, )
        return json.load(f)
