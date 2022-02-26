import os
import datetime
import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
FOLDER = 'data'
with open('./content_based_recsys.pkl', 'rb') as f:
    model = pickle.load(f)
train_df = pd.read_csv(os.path.join(FOLDER, 'train_df.csv'))
train_df[['contentId', 'personId']] = train_df[['contentId', 'personId']].astype(str)
items_df = pd.read_csv(os.path.join(FOLDER, 'shared_articles.csv'))
items_df['contentId'] = items_df['contentId'].astype(str)
def predict(user_id, topn):

    interacted_items = train_df[train_df['personId']==user_id]['contentId']
    itemid_to_ignore = set(interacted_items.values)
    recommend_df = model.recommend_items(user_id, itemid_to_ignore, topn=10000, verbose=False)
    itemid_to_recommend = set(recommend_df['contentId'].tolist())

    itemname_to_ignore = items_df[items_df['contentId'].isin(itemid_to_ignore)]['title'].tolist()
    itemname_to_recommend = items_df[items_df['contentId'].isin(itemid_to_recommend)]['title'].tolist()

    return itemname_to_ignore, itemname_to_recommend

@app.route('/inference', methods=['POST'])
def inference():

    params = request.get_json(force=True)

    # request parameters
    user_id = str(params['userid'])
    topn = params['topn']

    result = {
        'user_id': user_id,
        'topn': topn
    }

    try:
        itemname_to_ignore = predict(user_id, topn)
        itemname_to_ignore, itemid_to_recommend = predict(user_id, topn)
        result['articles_the_user_interacted'] = itemname_to_ignore
        result['articles_recommend_to_user'] = itemname_to_ignore
        result['status'] = 'success'
    except Exception as e:
        result['message'] = str(e)
        result['status'] = 'failed'

    result['timestamp'] = str(datetime.datetime.now())

    return jsonify(result)


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=int("5000"), debug=True) #map port to 5000