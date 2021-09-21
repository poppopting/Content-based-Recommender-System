import scipy
import sklearn
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from model.base import ContentBasedRecommender
class TfIdfBasedRec(ContentBasedRecommender):
    
    def __init__(self, item_df=None, 
                       ngram_range=(1, 2),
                       min_df=0.003,
                       max_df=0.5,
                       max_features=5000):

        self.item_df = item_df
        self.item_ids = item_df['contentId'].tolist()
        
        logger.info('Build tf-idf matrix')
        self.item_vec_matrix = self.build_tfidf(item_df, ngram_range, min_df, max_df, max_features)
    
    def build_tfidf(self, item_df, ngram_range, min_df, max_df, max_features):
           
        stopwords_list = stopwords.words('english')
        vectorizer = TfidfVectorizer(analyzer='word',
                             ngram_range=ngram_range,
                             min_df=min_df,
                             max_df=max_df,
                             max_features=max_features,
                             stop_words=stopwords_list)

        tfidf_matrix = vectorizer.fit_transform(self.item_df['title'] + " " + self.item_df['text'])

        return tfidf_matrix
    
