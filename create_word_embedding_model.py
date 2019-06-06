# Python Modules
import os, re
import warnings
warnings.filterwarnings('ignore')

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

import gensim
from gensim.models import Word2Vec
from gensim.similarities.index import AnnoyIndexer

# My Modules
from db import db_connect
from nlp_model_utils import preprocess

    
if __name__ == "__main__":
    # Constants    
    EMBEDDING_DIM = 100

    # Read in raw data
    engine = db_connect()
    Session = sessionmaker(bind=engine)
    session = Session()
    papers_df = pd.read_sql_table("paper", engine)
    print("Read data from DB.")

    porter = PorterStemmer()
    wordnet = WordNetLemmatizer()
    papers_df["tokens"] = papers_df.abstract.apply(lambda x: preprocess(porter, wordnet, x))
    print("Preprocessed abstracts.")

    model = gensim.models.Word2Vec(sentences=papers_df.tokens.tolist(), size=EMBEDDING_DIM, window=5, workers=4, min_count=1)
    print("Built word2vec embedding model.")

    words = list(model.wv.vocab)
    print(f"Vocabulary size: {len(words)}")

    out_filename = "abstract_embedding_word2vec.txt"
    model.wv.save_word2vec_format(out_filename)
    print(f"Saved embedding model to {out_filename}")

    indexer = AnnoyIndexer(model, 2)
    for word in ["gauss", "education", "machine", "learn", "robot"]:
        try:
            print(f"{word}: {model.most_similar(word, topn=5, indexer=indexer)}")
        except:
            pass