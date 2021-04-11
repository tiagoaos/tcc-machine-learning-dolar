import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
#from sklearn.externals import joblib
import matplotlib.pyplot as plt
import seaborn as sns
#import texthero as hero
from datetime import datetime
from nltk.corpus import stopwords
import nltk
from flask import Flask
import tweer
import tweepy
import json

app = Flask(__name__)
@app.route("/")

def hello():
    return "Hello World!"
from flask import make_response, jsonify


stock = {
    "fruit": {
        "apple": 30,
        "banana": 45,
        "cherry": 1000
    }
}

df = pd.read_csv('0dolar2020-11-01,2021-03-242.csv')
df2 = pd.read_csv('100dolar2020-11-01,2021-03-242.csv')
df3 = pd.read_csv('200dolar2020-11-01,2021-03-242.csv')


frames = [df, df2, df3]
pd_text = pd.concat(frames)
pd_text.rename(columns = {"published_at": "date"},inplace=True)
# Transforma a string em data mantendo somente a data
pd_text['date'] = pd.to_datetime(pd_text['date'], format='%Y-%m-%d')
pd_text['date2'] = pd_text['date'].dt.normalize()







df_dolar_real = pd.read_csv('USD_BRL.csv')

df_dolar_real.rename(columns = {"Último": "Ultimo"},inplace=True)
df_dolar_real.rename(columns = {"Var%": "Var"},inplace=True)
df_dolar_real.rename(columns = {"Data": "date"},inplace=True)

df_dolar_real['date'] = pd.to_datetime(df_dolar_real['date'], format='%d.%m.%Y')
df_dolar_real['date'] = df_dolar_real['date'].dt.normalize()

# df_dolar_real['Ultimo'] = df_dolar_real['Ultimo'].astype(int)
df_dolar_real = df_dolar_real.replace(',','.', regex=True)
df_dolar_real = df_dolar_real.replace('%','', regex=True)
df_dolar_real['Ultimo'] = pd.to_numeric(df_dolar_real['Ultimo'])
df_dolar_real['Máxima'] = pd.to_numeric(df_dolar_real['Máxima'])
df_dolar_real['Mínima'] = pd.to_numeric(df_dolar_real['Mínima'])


pd_text['date'] = pd.to_datetime(pd_text.date).dt.tz_localize(None)
pd_text['date'] = pd_text['date'].dt.normalize()
df_dolar_real.drop_duplicates()
pd_merge_media = pd.merge(pd_text[["title","date"]],df_dolar_real[['date','Var']])
pd_merge_media.head(10)
#define o var dolar
def var_dolar(fvar):
  ivar = 0
  ivar = float(fvar['Var'])
  if ivar>0:
    return 0
  else:
    return 1
pd_merge_media['var-dolar'] = pd_merge_media.apply(var_dolar, axis=1)
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')
# pd_merge_media['title'] = hero.remove_stopwords(pd_merge_media['title'], stopwords)
X = pd_merge_media['title']
y = pd_merge_media['var-dolar']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)


multinomial_clf = Pipeline([('cv', CountVectorizer(ngram_range=(1, 2))),                            
                     ('clf', MultinomialNB())])

complement_clf = Pipeline([('cv', CountVectorizer(ngram_range=(1, 2))),
                     ('clf', ComplementNB())])

svm_clf = Pipeline([('cv', CountVectorizer(ngram_range=(1, 2))),
                     ('clf', svm.SVC(kernel = 'linear'))])
multinomial_clf.fit(X_train, y_train)
complement_clf.fit(X_train, y_train)
svm_clf.fit(X_train, y_train)

pred_mult = multinomial_clf.predict(X_test)
pred_complement = complement_clf.predict(X_test)
pred_svm = svm_clf.predict(X_test)






# pd_text['pca'] = (
#    pd_text["title"]
#    .pipe(hero.clean)
#    .pipe(hero.tfidf)
#    .pipe(hero.pca)
# )

teste = ["dólar tem maior baixa semanal em 9 meses "]

# val_pred = multinomial_clf.predict(teste)
# print(val_pred)



# print(tweer.lasttweet())

@app.route("/dolar")
def get_stock():
    df = tweer.lasttweet()
    print(df.head(4))
    val_pred = multinomial_clf.predict(df['conteudo'])
    print(df)
    print(val_pred)
    type(val_pred)
    list_predict = np.array(val_pred).tolist()
    json_predict = json.dumps({"prediction": list_predict})
    json_predict
    res = make_response(str(json_predict), 200)
    return res

if __name__ == "__main__":
    app.run(port=80,host='0.0.0.0')

