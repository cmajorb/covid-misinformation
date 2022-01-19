import os
from os import walk
from os import listdir
from os.path import isdir, join
import csv
from collections import defaultdict

directories = [
'05-01-2020',
'07-01-2020',
'09-01-2020']
#'11-01-2020'

stats = {}
stats = defaultdict(lambda:0,stats)
claims = ['Fake','Real']
sources = ['Claim','News']
contents = ['tweets_replies','tweets','']
actual = {'Fake-News-tweets_replies': 7436, 'Fake-Claim-': 28, 'Fake-News-tweets': 10439, 'Real-Claim-tweets': 8092, 'Real-Claim-tweets_replies': 12451, 'Real-News-tweets': 141652, 'Fake-Claim-tweets': 484, 'Real-News-': 3565, 'Real-Claim-': 454, 'Fake-Claim-tweets_replies': 626, 'Real-News-tweets_replies': 114820, 'Fake-News-': 204}
mypath = '../../CoAID/'
for directory in directories:
    for filename in os.listdir(mypath + directory):
        name = ''
        for claim in claims:
            if claim in filename:
                name = name + claim + '-'
                break
        for source in sources:
            if source in filename:
                name = name + source + '-'
                break
        for content in contents:
            if content in filename:
                name = name + content
                break  
        file = open(mypath+directory+'/'+filename)
        reader = csv.reader(file)
        if(name=='Fake-News-' or name=='Real-News-'):
            lines = 0
            for row in reader:
                if row[1] == 'article':
                    lines += 1
        else:
            lines= len(list(reader)) - 1       
        stats[name] = stats[name] + lines
print(stats)
for i in actual:
    #print(i + ": " + str(100*(actual[i]-stats[i])/actual[i]))
    print(i + ": " + str(actual[i]-stats[i]) + " (%.2f%%)" % (100*(actual[i]-stats[i])/actual[i]))
