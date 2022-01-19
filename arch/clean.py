#! /usr/bin/python
# coding: utf-8

import re
import time
import gzip
import csv
import os
import sys
import string

keywords = ['gerardduncan63','araceliholden38']
with open('final.csv', 'rt') as f, open('final2.csv', 'w') as out:
    writer = csv.writer(out)
    original = 0
    deleted = 0
    reader = csv.reader((line.replace('\0','') for line in f), delimiter=",")
    for row in reader:
        original += 1
        total_count = []
        processed_string = row[2].lower().translate(str.maketrans(string.punctuation,' '*len(string.punctuation))).split()
        for keyword in keywords:
            count = processed_string.count(keyword)
            total_count.append(count)
        if sum(total_count) > 0:
            deleted = deleted + 1
        else:
            writer.writerow(row)
print(deleted)
print(original)

total = 0
with open('final2.csv','r') as in_file, open('final3.csv','w') as out_file:
    seen = set() # set for fast O(1) amortized lookup
    writer = csv.writer(out_file)
    for line in csv.reader(in_file):
        if line[2] in seen: continue # skip duplicate
        total += 1
        seen.add(line[2])
        writer.writerow(line)
print(total)
print(original-total)