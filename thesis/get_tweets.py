#from transformers import BertForSequenceClassification, BertTokenizer
#import torch
import pandas as pd
import requests
import os
import json
import sys
'''
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
'''

directories = [
'05-01-2020',
'07-01-2020',
'09-01-2020']
mypath = '../../CoAID/'

bearer_token = "AAAAAAAAAAAAAAAAAAAAAEU2YQEAAAAAg%2F8NHMK0lDOlNUhI%2FsJ1rSwlqwM%3DIEIWoJpbqOSL7aaNPajvClkFLY5ydQN1M35wUXvpVQQIXhGT4L"

def create_url(id):
    tweet_fields = "tweet.fields=text"
    # Tweet fields are adjustable.
    # Options include:
    # attachments, author_id, context_annotations,
    # conversation_id, created_at, entities, geo, id,
    # in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
    # possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
    # source, text, and withheld
    ids = "ids=" + id
    # You can adjust ids to include a single Tweets.
    # Or you can add to up to 100 comma-separated IDs
    url = "https://api.twitter.com/2/tweets?{}&{}".format(ids, tweet_fields)
    return url


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2TweetLookupPython"
    return r

def connect_to_endpoint(url):
    response = requests.request("GET", url, auth=bearer_oauth)
    #print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()

#data = json.dumps(json_response, indent=4, sort_keys=True)
'''
for tweet in json_response['data']:
    print(tweet['text'])
print(json_response['data'])
'''

def add_data(df,output,name,cont):
    if cont:
        final = pd.read_csv("./results/" + name)
    else:
        final = pd.DataFrame(columns=['tweet_id','input','output'])
    batch = ""
    print("reading " + str(df.shape[0]) + " tweets")
    for index, row in df.iterrows():
        if cont and index < 63099:
            continue
        batch += str(row['tweet_id']) + ","
        if((index+1)%100==0):
            print(index)
            url = create_url(batch[:-1])
            try:
                json_response = connect_to_endpoint(url)
                for tweet in json_response['data']:
                    final = final.append({"tweet_id":tweet['id'],"input":tweet['text'],"output":output},ignore_index=True)
            except:
                print("Failed")
                final.to_csv("./results/" + name)
                sys.exit()
            batch = ""
    url = create_url(batch[:-1])
    json_response = connect_to_endpoint(url)
    for tweet in json_response['data']:
        final = final.append({"tweet_id":tweet['id'],"input":tweet['text'],"output":output},ignore_index=True)
    '''
        tweet_id = str(row['tweet_id'])
        url = create_url(tweet_id)
        json_response = connect_to_endpoint(url)
        if 'data' in json_response:
            content = json_response['data'][0]['text']
            final.append([tweet_id,content,0])
    '''
    print(final.shape[0])
    final.to_csv("./results/" + name)

'''
final = pd.DataFrame(columns=['tweet_id','input','output'])
final.to_csv('final.csv')
'''

#0 = Fake, 1 = Real

for directory in directories:
    for filename in os.listdir(mypath + directory):
        name = directory+filename
        if name not in os.listdir("./results"):
            if "tweets.csv" in filename:
                print("checking " + mypath + directory+'/'+filename)
                if "Fake" in filename:
                    print("it's fake")
                    add_data(pd.read_csv(mypath+directory+'/'+filename),0,name,False)
                else:
                    print("it's real")
                    add_data(pd.read_csv(mypath+directory+'/'+filename),1,name,False)

'''
directory = "05-01-2020"
filename = "NewsRealCOVID-19_tweets.csv"
add_data(pd.read_csv(mypath+directory+'/'+filename),1,directory+filename,True)
'''