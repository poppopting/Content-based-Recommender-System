import scipy
import sklearn
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from model.base import ContentBasedRecommender
class Word2VecRec(ContentBasedRecommender):
    
    def __init__(self, item_df=None, 
                       size=250,
                       window=5,
                       min_count=5,
                       workers=4,
                       sg=1,
                       niter=20,
                       agg='mean',
                       ngram_range=(1, 2), 
                       max_df=0.5,
                       max_features=5000):

        self.item_df = item_df
        self.item_ids = item_df['contentId'].tolist()
        
        self.min_count = min_count
        logger.info('Build Word2Vec model')
        self.w2v = self.build_word2vec(item_df, size, window, min_count, workers, sg, niter)
        logger.info('Build item matrix by using Word2Vec model')
        self.item_vec_matrix = self.build_item_vec_matrix(self.w2v, size, agg, ngram_range, max_df, max_features)
    
    def build_word2vec(self, item_df, size, window, min_count, workers, sg, niter):
           
        text = self.item_df['title'] + " " + self.item_df['text']
        model = Word2Vec(text.apply(word_tokenize),
                         size=size,
                         window=window,
                         min_count=min_count,
                         workers=workers,
                         sg=sg,
                         iter=niter)

        return model
    
    def build_tfidf(self, item_df, ngram_range, max_df, max_features):
           
        stopwords_list = stopwords.words('english')
        self.vectorizer = TfidfVectorizer(analyzer='word',
                             ngram_range=ngram_range,
                             min_df=self.min_count,
                             max_df=max_df,
                             max_features=max_features,
                             stop_words=stopwords_list)

        tfidf_matrix = self.vectorizer.fit_transform(self.item_df['title'] + " " + self.item_df['text'])

        return tfidf_matrix
    
    def build_item_vec_matrix(self, model, size, agg='mean', ngram_range=(1, 2), max_df=0.5, max_features=5000):
        n_item = len(self.item_ids)
        matrix = scipy.sparse.csr_matrix((n_item, size))
        
        if agg != 'mean':
            logger.info('Build TF-IDF matrix as weight of word2vec ')
            tfidf_matrix = self.build_tfidf(self.item_df, ngram_range, max_df, max_features)
            vocab = self.vectorizer.get_feature_names()    
            article_vecs = np.stack([self.w2v.wv[word] if word in self.w2v.wv.vocab 
                                     else np.zeros(size)
                                     for word in vocab ])
        
        for i, item_id in enumerate(self.item_ids):
            
            if agg == 'mean':
                info = self.item_df[self.item_df['contentId']==item_id]
                tokens = word_tokenize(info['title'].values[0] + " " + info['text'].values[0])
                vec = [self.w2v.wv[word] for word in tokens if word in self.w2v.wv.vocab]

                item_vec = np.stack(vec).mean(axis=0)
            elif agg == 'tfidf':
                item_vec = np.average(article_vecs, weights=tfidf_matrix[i].toarray()[0], axis=0)
                
            else:
                raise Exception("Only support aggregation by 'mean' or 'tfidf'")
            matrix[i] = item_vec
            
        return matrix
    
    
class Doc2VecRec(ContentBasedRecommender):
    
    def __init__(self, item_df=None, 
                       size=250,
                       window=5,
                       min_count=5,
                       workers=4,
                       dm=1,
                       niter=20):

        self.item_df = item_df
        self.item_ids = item_df['contentId'].tolist()
        
        logger.info('Build Doc2Vec model')
        self.d2v = self.build_doc2vec(item_df, size, window, min_count, workers, dm, niter)
        logger.info('Build item matrix by using Doc2Vec model')
        self.item_vec_matrix = self.build_item_vec_matrix(self.d2v, size)
    
    def build_doc2vec(self, item_df, size, window, min_count, workers, dm, niter):
           
            
        tagged_article = [
            TaggedDocument(
                words=word_tokenize(p), tags=[i]) for i,p in enumerate(self.item_df['title'] + " " + self.item_df['text'])
        ]
        
        model = Doc2Vec(tagged_article,
                        size=size,
                        window=window,
                        min_count=min_count,
                        workers=workers,
                        dm=dm,
                        iter=niter)

        return model
    
    def build_item_vec_matrix(self, model, size):
        n_item = len(self.item_ids)
        matrix = scipy.sparse.csr_matrix((n_item, size))
        for i in range(n_item):
            matrix[i] = model.docvecs[i]
            
        return matrix
        