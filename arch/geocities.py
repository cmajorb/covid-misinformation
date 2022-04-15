import pandas as pd
import csv
import sys
import os

df = pd.read_csv("Geocities_Dataset.csv")
ids = df['ID'].tolist()
final = pd.DataFrame()
directory = '/mnt/storage/geocities/geocities-aut-csv-derivatives/webpages/'
csv.field_size_limit(sys.maxsize)
count = 0
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        location = directory + filename
        with open(location, 'rt') as f:
            reader = csv.reader((line.replace('\0','') for line in f), delimiter=",")
            for row in reader:
                try:
                    if row[0] in ids:
                        final = final.append({'id':row[0],'url':'https://web.archive.org/web/20091027/' + row[2], 'text':row[6]})
                        print(row[0])
                except ValueError:
                    pass

final.to_csv("final_text.csv")
                    
             