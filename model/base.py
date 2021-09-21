import scipy
import sklearn
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    
    def __init__(self, item_df=None, 
                       ngram_range=(1, 2),
                       min_df=0.003,
                       max_df=0.5,
                       max_features=5000):

        self.item_df = item_df
        self.item_ids = item_df['contentId'].tolist()
        
        self.item_vec_matrix = None 
    
    def item2vec(self, item_id):
        idx = self.item_ids.index(item_id)
        return self.item_vec_matrix[idx]
    
    def user2vec(self, user_df, user_id):
        one_user_df = user_df[user_df['personId']==user_id]
        interacted_items = one_user_df['contentId'].tolist()

        item_vecs = scipy.sparse.vstack(self.item2vec(item_id) for item_id in interacted_items)
        score = np.array(one_user_df['eventType']).reshape(-1,1)
        
        #Weighted average of item profiles by the interactions strength
        user_vec = np.sum(item_vecs.multiply(score), axis=0) / np.sum(score)
        user_vec = sklearn.preprocessing.normalize(user_vec)
        return user_vec
    
    def fit(self, user_df):

        self.user_profiles = {}
        logger.info('Begin computing user vectors by using item vector matrix')
        for user_id in user_df['personId'].unique():
            self.user_profiles[user_id] = self.user2vec(user_df, user_id)
        logger.info('Finish computing')
        return self
    
    def predict(self, user_id, topn=1000):
        #Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(self.user_profiles[user_id], self.item_vec_matrix)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = [(self.item_ids[i], cosine_similarities[0,i]) for i in similar_indices[::-1]]
        return similar_items
    
    
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self.predict(user_id)
        #Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        
        rec_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength']).head(topn)

        if verbose:
            rec_df = rec_df.merge(self.item_df, how = 'left', 
                                  left_on = 'contentId', 
                                  right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]

        return rec_df
