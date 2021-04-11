


import nltk
import tweepy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import tokens_tt
import re
from datetime import datetime
import pandas as pd

tweets = [] # Lista vazia para armazenar scores
tweets_text = []
twets_date = []
tweets_witout_url = []

currDate = datetime.now()

def lasttweet () :
        auth = tweepy.OAuthHandler(tokens_tt.consumer_key, tokens_tt.consumer_secret)
        auth.set_access_token(tokens_tt.access_token, tokens_tt.access_token_secret)
        api = tweepy.API(auth)  
        nltk.download('punkt')
        nltk.download('stopwords')

        stopwords = nltk.corpus.stopwords.words('portuguese')
        public_tweets = api.search('dolar lang:pt (from:InvestingBrasil OR from:valoreconomico OR from:UOLEconomia)',count =30)
        for tweet in public_tweets:
                tweet_tnk_date = []
                dt_obj  = ""
                without_url = re.sub(r'http\S+', "", tweet.text)
                without_url =re.sub(u'[^a-zA-Z0-9áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ: ]', '', without_url).lower()
                tokens = word_tokenize(without_url)
                filtered_sentence = [w for w in tokens if not w in stopwords]  
                if(datetime.now() - tweet.created_at ).days <3:
                        tweet_token = []
                        tweet_token.append(without_url)
                        tweet_token.append(tweet.created_at)
                        tweets_text.append(tweet_token)

        panda_tweet = pd.DataFrame(tweets_text)
        panda_tweet.rename(columns = {0: "conteudo"},inplace=True)
        panda_tweet.rename(columns = {1: "date"},inplace=True)
        panda_tweet['date'] = panda_tweet['date'].dt.normalize()
        return panda_tweet
             


