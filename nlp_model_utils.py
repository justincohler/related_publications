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

    papers_df["links_pctile"] = pd.qcut(papers_df.links, 100, labels=[i for i in range(100)])
    links = papers_df.links_pctile.values
    
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
    X_train = abstract_vectors_padded[:400]
    y_train = links[:400]
    X_test = abstract_vectors_padded[400:]
    y_test = links[400:]

    EMBEDDING_DIM = X_test.shape[1]

    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
    model.add(LSTM(500))
    model.add(Dense(100, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=32, epochs=25, validation_data=(X_test, y_test), verbose=2)