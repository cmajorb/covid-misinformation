#https://api.covidtracking.com/v1/states/info.json
#https://www.azdhs.gov/covid19/data/index.php
#Can also look at coverage (how long have these sites been around?)
import csv
import sys

csv.field_size_limit(sys.maxsize)
keywords = "https://www.azdhs.gov/covid19/data/index.php"

def getstuff(filename, criterion):
    with open(filename, "r") as csvfile:
        datareader = csv.reader(csvfile)
        yield next(datareader)  # yield the header row
        count = 0
        for row in datareader:
            print
            if criterion in row[1]:
                yield row
                count += 1
            elif count:
                # done when having read a consecutive series of rows 
                return

def getstuff(filename, criterion):
    with open(filename, "r") as csvfile:
        datareader = csv.reader(csvfile)
        yield next(datareader)  # yield the header row
        count = 0
        for row in datareader:
            print
            yield row
            count += 1


filename = '/mnt/storage/arch/html-file.csv'

import pandas as pd
chunksize = 10000
finaldf = pd.DataFrame()
for df in pd.read_csv(filename, chunksize=chunksize, iterator=True):
    df = df.loc[df['url'].str.contains("azdhs.gov", case=False)]
    if len(df) > 0:
        finaldf = pd.concat([finaldf,df])
        finaldf.to_csv("az2.csv")
        print(finaldf)
    break


for row in getstuff(filename,"azdhs.gov"):
    print(row[1])

#['crawl_date', 'url', 'filename', 'extension', 'mime_type_web_server', 'mime_type_tika', 'md5', 'sha1', 'content']


for row in getstuff(filename,""):
    print(row)
    break

from tableauscraper import TableauScraper as TS

url = "https://public.tableau.com/views/COVID-19Cases_15840488375320/COVID-19GlobalView"
ts = TS()
ts.loads(url)
workbook = ts.getWorkbook()

for t in workbook.worksheets:
    print(f"worksheet name : {t.name}") #show worksheet name
    print(t.data) #show dataframe for this worksheet



import pandas as pd
import time
start = time.time()
df = pd.read_csv(filename)
end = time.time()
print("Read csv without chunks: ",(end-start),"sec")

start = time.time()
#read data in chunks of 1 million rows at a time
chunk = pd.read_csv(filename,chunksize=1000000)
end = time.time()
print("Read csv with chunks: ",(end-start),"sec")
pd_df = pd.concat(chunk)



filename = '/mnt/storage/arch/html-file.csv'

def process(chunk):
    print("processing chunk")
    print(chunk)

import pandas as pd
chunksize = 10 ** 6
for chunk in pd.read_csv(filename, chunksize=chunksize):
    process(chunk)
