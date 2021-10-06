from rq import Queue, Worker
from redis import Redis
from processfile import process_file
from rq.registry import FailedJobRegistry, Job

import time
import os

# Tell RQ what Redis connection to use
redis_conn = Redis()
q = Queue('files',connection=redis_conn)  # no args implies the default queue
base = "/mnt/storage/geocities/geocities-aut-csv-derivatives/webpages/"



for filename in os.listdir(base):
    if filename.endswith(".csv"):
        job = q.enqueue(process_file, base + filename)
