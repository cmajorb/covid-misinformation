import pandas as pd
import requests
import os
import json
import sys


directories = [
'05-01-2020',
'07-01-2020',
'09-01-2020']
mypath = '../../CoAID/'

#0 = Fake, 1 = Real

final = pd.DataFrame(columns=['input', 'output'])
for directory in directories:
    for filename in os.listdir(mypath + directory):
        if "NewsFakeCOVID-19.csv" in filename:
            print("checking " + mypath + directory+'/'+filename)
            df_in = pd.read_csv(mypath + directory+'/'+filename)
            df = pd.DataFrame()
            df['input'] = df_in['content']
            df['output'] = 1
            final = pd.concat([final, df])
        elif "NewsRealCOVID-19.csv" in filename:
            print("checking2 " + mypath + directory+'/'+filename)
            df_in = pd.read_csv(mypath + directory+'/'+filename)
            df = pd.DataFrame()
            df['input'] = df_in['content']
            df['output'] = 0
            final = pd.concat([final, df])
