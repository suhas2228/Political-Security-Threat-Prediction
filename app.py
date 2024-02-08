from random import random
from xml.parsers.expat import model
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import numpy as np
import nltk
from sklearn.datasets import load_files
#nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import string 
from nltk.stem import WordNetLemmatizer
import sqlite3
from googletrans import Translator
import warnings
from topic_modelling import Topic_modeling
import pandas as pd
import sqlite3
import random

import smtplib 
from email.message import EmailMessage
from datetime import datetime


app = Flask(__name__)

translator = Translator()

#Reading data

df= pd.read_csv('processed.csv',encoding='ISO-8859-1')
df = df[0:1000]

Tweet = []
Labels = []

for row in df["message"]:
    #tokenize words
    words = word_tokenize(row)
    #remove punctuations
    clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
    #remove stop words
    english_stops = set(stopwords.words('english'))
    characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
    clean_words = [word for word in clean_words if word not in english_stops]
    clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
    #Lematise words
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    Tweet.append(lemma_list)



#df['message']=df['text']

# Import label encoder 
from sklearn import preprocessing 

# label_encoder object knows 
# how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 

# Encode labels in column 'species'. 
df['Sentiment']= label_encoder.fit_transform(df['Sentiment']) 


#df = df[0:2000]
X = df['message']
y = df['Sentiment']

# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier

estimators = [('rf', RandomForestClassifier(n_estimators=10)),('mlp', MLPClassifier(random_state=1, max_iter=30))]

clf = StackingClassifier(estimators=estimators, final_estimator=LGBMClassifier())
clf.fit(X,y)



@app.route('/')
def home():
	return render_template('home.html')




@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        
        message = request.form['message']
        translations = translator.translate(message, dest='en')
        message =  translations.text
        data = [message]
        #cv = CountVectorizer()
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        
        print(my_prediction[0])
        df = pd.DataFrame({'sentence':data})
        t,word = Topic_modeling(df)

        return render_template('result.html',prediction = my_prediction,message=message,to = t, wo = word)


@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')


@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "evotingotp4@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("evotingotp4@gmail.com", "xowpojqyiygprhgr")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict1', methods=['POST'])
def predict1():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signin.html")



@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/notebook')
def notebook():
	return render_template('Notebook.html')


if __name__ == '__main__':
	app.run(debug=False)
