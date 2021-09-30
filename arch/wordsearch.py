
import csv
import sys
import os

directory = '/mnt/storage/geocities/geocities-aut-csv-derivatives/webpages/'
csv.field_size_limit(sys.maxsize)
count = 0
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        location = directory + filename
        with open(location, 'rt') as f:
            #reader = csv.reader(f, delimiter=',')
            reader = csv.reader((line.replace('\0','') for line in f), delimiter=",")
            for row in reader:
                for field in row:
                    try:
                        loc = field.index(' hiv ')
                    except ValueError:
                        pass
                    else:
                        count = count + 1
    if count%10 == 0:
        print(count)
