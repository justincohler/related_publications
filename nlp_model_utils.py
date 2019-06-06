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
from keras.layers import Dense, Embedding, LSTM, GRU, Flatten, Reshape, Activation, Dropout
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


# Constants    
EMBEDDING_DIM = 100

def preprocess(porter: PorterStemmer, wordnet: WordNetLemmatizer, sentence: str) -> list:
    """Return a stemmed, lemmatized, tokenized list of words for a given sentence.
    
    Arguments:
        porter {PorterStemmer} -- The Porter Stemmer
        wordnet {WordNetLemmatizer} -- WordNet Lemmatizer
        sentence {str} -- Input sentence
    
    Returns:
        list -- A list of cleaned words
    """
    sentence = re.sub('[!?:.,;@#$]', '', sentence)
    words = nltk.word_tokenize(sentence)
    return [wordnet.lemmatize(word) for word in words]

def import_word_embedding_model(filename: str) -> dict:
    """Return word embedding dictionary from file.
    
    Arguments:
        filename {str} -- The filename in the current directory
    
    Returns:
        dict -- The word embedding dict to return
    """
    embeddings = {}
    with open(os.path.join('', filename), encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:])
            embeddings[word] = coefs
    return embeddings  
    
if __name__ == "__main__":

    # Read word embeddings
    in_filename = "abstract_embedding_word2vec.txt"
    embeddings = import_word_embedding_model(in_filename)

    # Read in raw data
    engine = db_connect()
    Session = sessionmaker(bind=engine)
    session = Session()
    papers_df = pd.read_sql_table("paper", engine)

    porter = PorterStemmer()
    wordnet = WordNetLemmatizer()
    papers_df["tokens"] = papers_df.abstract.apply(lambda x: preprocess(porter, wordnet, x))
    print("Preprocessed abstracts.")


    # Attribution to https://towardsdatascience.com/machine-learning-word-embedding-sentiment-classification-using-keras-b83c28087456
    tokenizer = Tokenizer()
    abstract_lines = papers_df.tokens.tolist()
    tokenizer.fit_on_texts(abstract_lines)
    sequences = tokenizer.texts_to_sequences(abstract_lines)
    
    # Pad sequences
    print(f"Found {len(tokenizer.word_index)} unique tokens.")
    max_length = max([len(s) for s in abstract_lines])
    abstract_vectors_padded = pad_sequences(sequences, maxlen=max_length)

    BUCKETS = 100
    papers_df["links_pctile"] = pd.qcut(papers_df.links, BUCKETS, labels=[i for i in range(BUCKETS)])
    links = papers_df.links_pctile.values
    # links = papers_df.links.values
    
    print(f"Shape of abstract tensor: {abstract_vectors_padded.shape}")
    print(f"Shape of links tensor: {links.shape}")

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

    for word, i in tokenizer.word_index.items():
        if i > vocab_size:
            continue
        if word in embeddings:
            embedding_matrix[i] = embeddings[word]

    print(vocab_size)

    train_test_split = 0.7
    split_index = int(len(links)*train_test_split)

    X_train = abstract_vectors_padded[:split_index]
    y_train = links[:split_index]
    X_test = abstract_vectors_padded[split_index:]
    y_test = links[split_index:]

    EMBEDDING_DIM = X_train.shape[1]

    model_cont = Sequential()
    model_cont.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_length))
    model_cont.add(Flatten())
    model_cont.add(Dense(512, activation='relu'))
    # model_cont.add(Dropout(0.5))
    model_cont.add(Dense(256, activation='relu'))
    # model_cont.add(Dropout(0.5))
    model_cont.add(Dense(128, activation='relu'))
    # model_cont.add(Dropout(0.5))
    model_cont.add(Dense(64, activation='relu'))
    # model_cont.add(Dropout(0.5))
    model_cont.add(Dense(1))
    print(model_cont.summary())

    model_cont.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    model_cont.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), verbose=2)



    # Categorical Model
    # model_cat = Sequential()
    # model_cat.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_length))
    # model_cat.add(Flatten())
    # model_cat.add(Dense(512, activation='relu'))
    # model_cat.add(Dropout(0.3))
    # model_cat.add(Dense(256, activation='relu'))
    # model_cat.add(Dropout(0.3))
    # model_cat.add(Dense(128, activation='relu'))
    # model_cat.add(Dropout(0.3))
    # model_cat.add(Dense(64, activation='relu'))
    # model_cat.add(Dropout(0.3))
    # model_cat.add(Dense(1))
    # print(model_cat.summary())

    # model_cat.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
