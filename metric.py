import numpy as np
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, train):
        self.train = train
    
    def get_items_interacted(self, user_id):
        interacted_items = self.train[self.train['personId']==user_id]['contentId']
        
        return set(interacted_items.values)
    
    def get_ordered_recommend_items(self, model, user_id):
        items_to_ignore = self.get_items_interacted(user_id)
        recommend_df = model.recommend_items(user_id, items_to_ignore, topn=10000, verbose=False)
        
        return recommend_df['contentId'].tolist()
    
    def topn_precision_recall(self, pred, target, topn=10):
        topn_pred = np.asarray(pred[:topn])
        topn_target = np.asarray(target[:topn])

        hit_items = set(topn_pred).intersection(set(target))
        n_hit = len(hit_items)
        precision = n_hit / topn
        recall = n_hit / len(target)

        return n_hit, precision, recall
        
    def evaluate_user(self, model, user_id, test):
        pred = self.get_ordered_recommend_items(model, user_id)
        target =  test[test['personId']==user_id]['contentId'].tolist()
        
        n_hit05, precision05, recall05 = self.topn_precision_recall(pred, target, topn=5)
        n_hit10, precision10, recall10 = self.topn_precision_recall(pred, target, topn=10)
        n_hit20, precision20, recall20 = self.topn_precision_recall(pred, target, topn=20)
        
        return len(target), n_hit05, precision05, recall05, n_hit10, precision10, recall10, n_hit20, precision20, recall20
                
    def evaluate(self, model, test):
        
        all_user = []
        for user_id in test['personId'].unique():
            n_targ, n_hit05, pre05, re05, n_hit10, pre10, re10, n_hit20, pre20, re20 = self.evaluate_user(model, user_id, test)
            all_user.append([user_id, n_targ, 
                             n_hit05, pre05, re05,
                             n_hit10, pre10, re10,
                             n_hit20, pre20, re20])
            
        all_user = pd.DataFrame(all_user, columns=['user_id', 'num_interacted_items',
                                                   'num_hit@05', 'precision@05', 'recall@05',
                                                   'num_hit@10', 'precision@10', 'recall@10',
                                                   'num_hit@20', 'precision@20', 'recall@20'])
        
        num_all_interacted_items = all_user['num_interacted_items'].sum()
        global_metric = {}
        for num_hit in ['num_hit@05', 'num_hit@10', 'num_hit@20']:
            topn = int(num_hit[-2:])
            global_num_hit = all_user[num_hit].sum() 
            global_metric[f'precision@{topn:02d}'] = round(global_num_hit / (topn * all_user.shape[0]), 4)
            global_metric[f'recall@{topn:02d}'] = round(global_num_hit / num_all_interacted_items, 4)
            
        return all_user, global_metric
