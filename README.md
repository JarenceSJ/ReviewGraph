# Review Graph
## This code is under tidying up ... 

## requirements

```
python=3.8.3
transformers=3.1.0
dgl == 0.7.2
pytorch == 1.10.2
```

## Data prepration

1. Run word2vector.py for word embedding. Glove pretraining weight is required. The word embedding is insufficent for RGCL but legacy for other review-based recommendation models. 
2. Make sure can run load_sentiment_data in load_data.py 
3. Run BERT/bert_whitening.py for obtaining the feature vector for each review.
4. If previous steps successfully run, then you can run rgc_nd_ed.py. 


## Files
```
ReviewGraph
├── BERT
│ └── bert_whitening.py   # bert-whitening 
├── README.md
├── RGCL                  # models
│ ├── data.py             # load data by dgl
│ ├── rgc.py              # review graph learning 
│ ├── rgc_ed.py           # review graph learning with edge discrimination
│ ├── rgc_nd.py           # review graph learning with node discrimination
│ └── rgc_nd_ed.py        # RGCL
├── load_data.py		  
├── nlp_util.py           # clean text
├── util.py				
└── word2vector.py        # loading Glove pretraining word vectors.
```
