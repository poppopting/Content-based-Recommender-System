import os
import pandas as pd
from metric import ModelEvaluator

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from model.tfidf_based import TfIdfBasedRec
from model.vector_based import Word2VecRec, Doc2VecRec

def main():
    
    FOLDER = 'data'

    logger.info('Read data')
    train_df = pd.read_csv(os.path.join(FOLDER, 'train_df.csv'))
    test_df = pd.read_csv(os.path.join(FOLDER, 'test_df.csv'))

    articles_df = pd.read_csv(os.path.join(FOLDER, 'simplified_articles.csv'))

    logger.info('Init model and fit')
#     cbrec = TfIdfBasedRec(articles_df)
#     cbrec = Word2VecRec(articles_df, agg='mean')
#     cbrec = Word2VecRec(articles_df, agg='tfidf')
    cbrec = Doc2VecRec(articles_df)
    cbrec.fit(train_df)

    logger.info('Evaluate model')

    model_evaluator = ModelEvaluator(train_df)
    all_user, global_metric = model_evaluator.evaluate(cbrec, test_df)

    logger.info(f'{global_metric}')

if __name__ == '__main__':
    main()
