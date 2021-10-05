#! /usr/bin/python
# coding: utf-8

import re
import time
import gzip
import csv
import os


def process_file(file):
    csv.field_size_limit(sys.maxsize)
    filename = os.path.basename(file)
    with open(file, 'rt') as f:
            reader = csv.reader((line.replace('\0','') for line in f), delimiter=",")
            for row in reader:
                try:
                    loc = row[6].index(' hiv ')
                    with open("/home/cmbrow38/arch/results/"+ filename + ".txt", "w") as f:
                        f.write(row[0] + ','  + row[2] + ',' + row[6][loc-50:loc+50] + "\n")
                except ValueError:
                    pass
    
    




    with open('results/allaccounts.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                susaccounts.add(int(row['userid']))
            except Exception as e:
                print(e)
                pass
    base = '/mnt/storage/ows_raw/'
    count1 = 0
    count2 = 0
    count3 = 0
    with gzip.open(base + filename,'rb') as f:
        for line in f:
            line = line.strip()
            try:
                tweet = simplejson.loads(line)
                if ("entities" in tweet and "text" in tweet):
                    count1 = count1 + 1
                    if tweet["user"]["id"] in susaccounts:
                        count2 = count2 + 1
                    for mention in tweet["entities"]["user_mentions"]:
                        if mention["id"] in susaccounts:
                            count3 = count3 + 1
            except Exception as e:

