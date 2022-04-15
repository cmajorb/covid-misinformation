import json
import sys
import string
import re

path = "/mnt/storage/covid19/"
#filename="data"+sys.argv[1]+"2012.json"
filename = "stream_covid19_2020090115.json"
line_generator = open(path + filename)
output = open("/home/cmbrow38/covid_tweets.csv","wt")
for line in line_generator:
    try:
        line_object = json.loads(line)
        original_text = line_object["extended_tweet"]["full_text"]
        clean_text = original_text.lower()
        clean_text = clean_text.replace('[^\w\s]','')
        clean_text = ''.join([str(char) for char in clean_text if char in string.printable])
        clean_text = re.sub(r'http\S+', '', clean_text)
        clean_text = clean_text.translate(str.maketrans('', '', string.punctuation))
        clean_text = clean_text.replace('\n', ' ').replace('\r', '')
        output.write('"' + clean_text + '"' + '\n')
    except Exception as e:
        pass
