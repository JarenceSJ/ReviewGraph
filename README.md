# Review Graph

This repo is an official implementation of [A Review-aware Graph Contrastive Learning Framework for Recommendation](https://dl.acm.org/doi/abs/10.1145/3477495.3531927).

## Requirements

```
python == 3.8.3
transformers == 3.1.0
dgl == 0.7.2
pytorch == 1.10.2
```

## Data prepration

1. Run word2vector.py for word embedding. Glove pretraining weight is required. The word embedding is insufficent for RGCL but legacy for other review-based recommendation models. 
2. Make sure can run load_sentiment_data in load_data.py 
3. Run BERT/bert_whitening.py for obtaining the feature vector for each review.
4. If previous steps successfully run, then you can run rgc_nd_ed.py. 

### A processed data: Digital_Music
Dowload from [here](https://drive.google.com/drive/folders/1OPkb_XLlxDp4otLy5-WKX4j_RvxODPuj?usp=sharing).
Then config dataset_path parameter in rgc.py and the review_feat_path parameter in data.py.

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
