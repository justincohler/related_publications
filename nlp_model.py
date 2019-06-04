# Python Modules
import os, math, random, copy, re
import warnings
warnings.filterwarnings('ignore')
from collections import Counter, defaultdict

# SQL Modules
from sqlalchemy.orm import sessionmaker

# Scientific Modules
import pandas as pd
pd.set_option('max_colwidth', 20000)
import numpy as np

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.wrappers.scikit_learn import KerasRegressor
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import gensim
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import common_texts

# My Modules
import scholar
from db import db_connect


def lemmatize(wordnet, sentence):
    sentence = re.sub('[!?:.,;@#$]', '', sentence)
    words = nltk.word_tokenize(sentence)
    
    new_sentence = ""
    for word in words:
        new_sentence += wordnet.lemmatize(word) + " "
    
    return new_sentence.strip()
    
if __name__ == "__main__":

    # nltk.download('punkt')

    # Read in raw data
    engine = db_connect()
    Session = sessionmaker(bind=engine)
    session = Session()
    papers_df = pd.read_sql_table("paper", engine)

    # Preprocess text
    porter = PorterStemmer()
    wordnet = WordNetLemmatizer()

    papers_df["cleaned"] = papers_df.abstract.apply(lambda x: porter.stem(x))
    papers_df.cleaned = papers_df.cleaned.apply(lambda x: lemmatize(wordnet, x))

    # Attribution to https://towardsdatascience.com/machine-learning-word-embedding-sentiment-classification-using-keras-b83c28087456
    X_train = papers_df.loc[:400, 'cleaned'].values
    y_train = papers_df.loc[:400, 'links'].values
    X_test = papers_df.loc[400:, 'cleaned'].values
    y_test = papers_df.loc[400:, 'links'].values

    # Transform into Word Embedding
    tokenizer = Tokenizer()
    all_abstracts = np.hstack((X_train, X_test))
    tokenizer.fit_on_texts(all_abstracts)

    max_length = max([len(s.split()) for s in all_abstracts])

    vocab_size = len(tokenizer.word_index) + 1

    # Tokenize
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Pad
    X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
    X_test = pad_sequences(X_test, maxlen=max_length, padding='post')

    EMBEDDING_DIM = X_test.shape[1]

    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
    model.add(GRU(units=32, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=32, epochs=25, validation_data=(X_test, y_test), verbose=2)