import sqlite3
import pandas as pd
import tweepy as tw

# Execute a given SQL query on the database and return values
def execute_query(path, query):
    conn = sqlite3.connect(path)
    # execute query on database and retrieve them with fetchall
    results = conn.cursor().execute(query).fetchall()
    return results


# Execute query and load results into pandas dataframe
def execute_query_pandas(path, query):
    conn = sqlite3.connect(path)
    df = pd.read_sql_query(query, conn)
    return df



db_path = '/home/cmbrow38/databases/nela-covid-2021.db'

source = "thenewyorktimes"
query = "SELECT id, article_id, embedded_tweet FROM tweet"

id = "drudgereport--2021-01-01--SUPER-STRAIN TSUNAMI?"
query = "SELECT * FROM newsdata WHERE id='%s'" % id
query = "SELECT substr(article_id, 0, INSTR(article_id,'--')) AS source, tweet_id FROM tweet"


df = execute_query_pandas(db_path, query)
df.dropna(subset=['tweet_id'], inplace=True)
df = df[df.tweet_id != '']

patternDel = "\d{19}"
filter = df['tweet_id'].str.contains(patternDel)
df = df[filter]
df['tweet_id'] = df['tweet_id'].str.extract('(\d+)', expand=False)
df["tweet_id"] = pd.to_numeric(df["tweet_id"])
df = df.drop_duplicates()



tweet_df = pd.read_csv('twitter_data.csv')
tweet_df = tweet_df.rename(columns={"id": "tweet_id"})
tweet_df = tweet_df.drop(['Unnamed: 0'], axis=1)
tweet_df = tweet_df.drop_duplicates()

df_combined = pd.merge(tweet_df,df,on='tweet_id')
#What to do with tweets that are referenced in different outlets?

labels = pd.read_csv('/home/cmbrow38/databases/labels_all.csv')
labels = labels[labels.label != -1]
labels = labels[['source','label']]

df_combined = df_combined.merge(labels, how='left', on='source')
df_combined = df_combined.dropna()
df_combined = df_combined[['text','label']]
df_combined = df_combined.rename(columns={"text":"input", "label":"output"})
df_combined = df_combined.groupby('input').agg('max')
df_combined.to_csv('train.csv')

# Reliable - class 0
# Unreliable - class 1
# Mixed - class 2


'''
my_api_key = "x6N8m5XT9Z9Ii2LISyKPsey8g"
my_api_secret = "FiIf3ICxoJYZ1EqJPjjdAxEy1X77bIlxUYHoqkRLL0YjyTrR66"
# authenticate
auth = tw.OAuthHandler(my_api_key, my_api_secret)
api = tw.API(auth, wait_on_rate_limit=True)

id_list = []
final = pd.DataFrame()
for i in df.index:
    if (i+1)%100 == 0:
        try:
            tweets_fetched = api.lookup_statuses(id_list, tweet_mode = "extended")
            for tweet in tweets_fetched:
                tweet_df = {'id': str(tweet.id), 'text':tweet.full_text}
                final = final.append(tweet_df, ignore_index = True)
            print("fetched " + str(i))
        except Exception as e:
            print(e + str(i))
            continue
        id_list = []
    else:
        id_list.append(df.at[i, "tweet_id"])

final.to_csv('twitter_data.csv')
'''