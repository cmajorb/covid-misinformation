#! /usr/bin/python
# coding: utf-8
'''
from processfile import process_file
process_file('/mnt/storage/geocities/geocities-aut-csv-derivatives/webpages/part-04439-325f80ee-3749-4d6f-baec-3dc32abfb75d-c000.csv')
'''
import pandas as pd

def process_file(df,i):
    df = df.loc[df['url'].str.contains("azdhs.gov", case=False)]
    if len(df) > 0:
        df.to_csv("/home/cmbrow38/covid-misinformation/arch/covid/results/az"+str(i)+".csv")