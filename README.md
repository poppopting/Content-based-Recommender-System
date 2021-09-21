# Content-based-Recommender-System
### Abstract : 
This repo implement Content-based Recommender System by using TF-IDF, Word2Vec, Doc2Vec  
- ref: https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101

### Dataset :
- kaggle: [Articles sharing and reading from CI&T DeskDrop](https://www.kaggle.com/gspmoreira/articles-sharing-reading-from-cit-deskdrop)

### Method : 
* TF-IDF based
* Word2Vec based
  - Average vectors of words in article to obtain vector for each article
  - Weighted avergae vectors of words in article by taking word's TF-IDF value as weights to obtain vector for each article
* Doc2Vec based
