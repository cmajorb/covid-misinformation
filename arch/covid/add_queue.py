from rq import Queue, Worker
from redis import Redis
from processfile import process_file
from rq.registry import FailedJobRegistry, Job

import time
import os

#delete previous files
# dir = '/home/cmbrow38/covid-misinformation/arch/results'
# for f in os.listdir(dir):
#     os.remove(os.path.join(dir, f))

# Tell RQ what Redis connection to use
redis_conn = Redis()
q = Queue('files',connection=redis_conn)  # no args implies the default queue

filename = '/mnt/storage/arch/html-file.csv'

import pandas as pd
chunksize = 10000
i = 0
for df in pd.read_csv(filename, chunksize=chunksize, iterator=True):
    job = q.enqueue(process_file, df, i)
    i += 1
    print(i)


# import csv
# reader = csv.reader(open(filename, 'rb'))

# chunk, chunksize = [], 1000

# for i, line in enumerate(reader):
#     if (i % chunksize == 0 and i > 0):
#         q.enqueue(process_file, chunk)
#         del chunk[:]  # or: chunk = []
#     chunk.append(line)

# # process the remainder
# q.enqueue(process_file, chunk)


'''
import gzip
import io 

def get_csv_data_raw2(filename):
    data = []
    with open(filename, 'rb') as r_file:
        f = io.BufferedReader(r_file)
        for line in f:
            if b'azdhs.gov' in line:
                data.append(line)
                print(line)
    return data

data = get_csv_data_raw2('/mnt/storage/arch/html-file.csv')
'''