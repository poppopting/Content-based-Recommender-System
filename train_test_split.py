import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def data_split(folder, num_interact=5, test_size=0.2):
    article_df = pd.read_csv(os.path.join(folder, 'shared_articles.csv'))
    interact_df = pd.read_csv(os.path.join(folder, 'users_interactions.csv'))
    
    logger.info('Only keep English articles')
    article_df  = article_df[(article_df['lang']=='en') & (article_df['eventType'] == 'CONTENT SHARED')]
#     article_df.to_csv(os.path.join(folder, 'simplified_articles.csv'), index=False)
    interact_df = interact_df.loc[interact_df['contentId'].isin(article_df['contentId']), ['eventType', 'contentId', 'personId']]

    logger.info('Conver action type to score')
    action_score = {
        'VIEW': 1,
        'LIKE': 2,           
        'BOOKMARK': 3,      
        'COMMENT CREATED': 4,    
        'FOLLOW': 5           
    }
    logger.info(f'{action_score}')
    interact_df['eventType'] = interact_df['eventType'].map(action_score)

    interact_df = interact_df.groupby(['contentId', 'personId'])[['eventType']].sum().reset_index()
    
    logger.info(f'Only keep the user who has more than {num_interact} interactions')
    user, cnt = np.unique(interact_df['personId'], return_counts=True)
    user_with_enough_interact = user[cnt>=num_interact]
    interact_df = interact_df[interact_df['personId'].isin(user_with_enough_interact)]

    logger.info(f'Train test split by size = {test_size}')
    train_df, test_df = train_test_split(interact_df,
                                         stratify=interact_df['personId'], 
                                         test_size=test_size,
                                         random_state=42)
    
    logger.info('Complete train test split')
    return train_df, test_df

if __name__ == '__main__':

    FOLDER = 'data'
    NUM_INTERACT = 5
    TEST_SIZE = 0.2

    train_df, test_df = data_split(FOLDER, NUM_INTERACT, TEST_SIZE)

    train_df.to_csv(os.path.join(FOLDER, 'train_df.csv'), index=False)
    test_df.to_csv(os.path.join(FOLDER, 'test_df.csv'), index=False)

