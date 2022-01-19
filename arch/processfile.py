#! /usr/bin/python
# coding: utf-8

import re
import time
import gzip
import csv
import os
import sys
import string

'''
from processfile import process_file
process_file('/mnt/storage/geocities/geocities-aut-csv-derivatives/webpages/part-04439-325f80ee-3749-4d6f-baec-3dc32abfb75d-c000.csv')
'''
def process_file(filepath):
    csv.field_size_limit(sys.maxsize)
    filename = os.path.basename(filepath)
    keywords = ['aids', 'hiv', 'grid','hiv/aids','virus']
    #GRID -> gay, queer
    with open(filepath, 'rt') as f:
        reader = csv.reader((line.replace('\0','') for line in f), delimiter=",")
        for row in reader:
            try:
                total_count = []
                processed_string = row[6].lower().translate(str.maketrans(string.punctuation,' '*len(string.punctuation))).split()
                for keyword in keywords:
                    count = processed_string.count(keyword)
                    total_count.append(count)
                if sum(total_count) > 0:
                    with open("/home/cmbrow38/covid-misinformation/arch/results/"+ filename, "a") as output:
                        output.write(row[0] + ',"https://web.archive.org/web/20091027*/'  + row[2] + '",' + ','.join(str(e) for e in total_count) +  "\n")
            except ValueError:
                pass

