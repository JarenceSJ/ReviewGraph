"""MovieLens dataset"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch as th
import dgl
import sys

sys.path.append("..")

# from load_data import load_sentiment_data, \
#     load_data_for_review_based_rating_prediction
from load_data import *
from util import *


# _urls = {
#     'ml-100k' : 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
#     'ml-1m' : 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
#     'ml-10m' : 'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
# }
#
# READ_DATASET_PATH = get_download_dir()
# GENRES_ML_100K =\
#     ['unknown', 'Action', 'Adventure', 'Animation',
#      'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
#      'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
#      'Thriller', 'War', 'Western']
# GENRES_ML_1M = GENRES_ML_100K[1:]
# GENRES_ML_10M = GENRES_ML_100K + ['IMAX']


class MovieLens(object):

    def __init__(self, dataset_name, dataset_path, device, review_fea_size,
                 symm=True, mix_cpu_gpu=True, use_user_item_doc=False):
        self._device = device
        self._review_fea_size = review_fea_size
        self._symm = symm

        review_feat_path = \
            f'../checkpoint/{dataset_name}/BERT-Whitening/bert-base-uncased_sentence_vectors_dim_{review_fea_size}.pkl'

        try:
            self.train_review_feat = torch.load(review_feat_path)
        except FileNotFoundError:
            self.train_review_feat = None
            print(f'Load pretrained review feature fail! feature size:{review_fea_size}')

        if 'ml-100k' in dataset_path:
            sent_train_data, sent_valid_data, sent_test_data, _, _, dataset_info = \
                self.load_ml100k(dataset_path)
        else:
            sent_train_data, sent_valid_data, sent_test_data, _, _, dataset_info = \
                load_sentiment_data(dataset_path)

        if use_user_item_doc:
            doc_data = load_data_for_review_based_rating_prediction(dataset_path)
            self.word2id = doc_data['word2id']
            self.embedding = doc_data['embeddings']
            self.user_doc = torch.from_numpy(process_doc(doc_data['user_doc'], self.word2id))
            self.movie_doc = torch.from_numpy(process_doc(doc_data['item_doc'], self.word2id))
            if not mix_cpu_gpu:
                self.user_doc.to(device)
                self.movie_doc.to(device)
        else:
            self.word2id = None
            self.embedding = None
            self.user_doc = None
            self.movie_doc = None

        def process_sent_data(info):
            user_id = info['user_id'].to_list()
            item_id = info['item_id'].to_list()
            rating = info['rating'].to_list()

            return user_id, item_id, rating

        self.train_datas = process_sent_data(sent_train_data)
        self.valid_datas = process_sent_data(sent_valid_data)
        self.test_datas = process_sent_data(sent_test_data)
        self.possible_rating_values = np.unique(self.train_datas[2])

        self._num_user = dataset_info['user_size']
        self._num_movie = dataset_info['item_size']

        self.user_feature = None
        self.movie_feature = None

        # Generate features
        # if review_fea_size == 0:
        #     self.user_feature = None
        #     self.movie_feature = None
        #     # self.user_feature_groupby_rating = {i: None for i in self.possible_rating_values}
        #     # self.movie_feature_groupby_rating = {i: None for i in self.possible_rating_values}
        # else:
        #     # if mix_cpu_gpu, we put features in CPU
        #     self.user_feature, self.movie_feature = self._process_user_item_review_feat()
        #     # self.user_feature_groupby_rating, self.movie_feature_groupby_rating = self._process_user_item_review_feat_groupby_rating()
        #     if not mix_cpu_gpu:
        #         self.user_feature = self.user_feature.to(self._device)
        #         self.movie_feature = self.movie_feature.to(self._device)
        #         # self.user_feature_groupby_rating = {k: v.to(self._device) for k, v in self.user_feature_groupby_rating.items()}
        #         # self.movie_feature_groupby_rating = {k: v.to(self._device) for k, v in self.movie_feature_groupby_rating.items()}

        self.user_feature_shape = (self.num_user, self.num_user)
        self.movie_feature_shape = (self.num_movie, self.num_movie)

        # self.user_feature_shape = (self.num_user, self.num_user)
        # self.movie_feature_shape = (self.num_movie, self.num_movie)

        info_line = "Feature dim: "
        info_line += "\nuser: {}".format(self.user_feature_shape)
        info_line += "\nmovie: {}".format(self.movie_feature_shape)
        print(info_line)

        # all_train_rating_pairs, all_train_rating_values = self._generate_pair_value('all_train')
        train_rating_pairs, train_rating_values = self._generate_pair_value('train')
        valid_rating_pairs, valid_rating_values = self._generate_pair_value('valid')
        test_rating_pairs, test_rating_values = self._generate_pair_value('test')

        def _make_labels(ratings):
            """
            不同rating值对应id
            """
            labels = th.LongTensor(
                np.searchsorted(self.possible_rating_values, ratings)).to(
                device)
            return labels

        self.train_enc_graph = self._generate_enc_graph(train_rating_pairs,
                                                        train_rating_values,
                                                        add_support=True)
        self.train_dec_graph = self._generate_dec_graph(train_rating_pairs,
                                                        review_feat=self.train_review_feat)
        self.train_labels = _make_labels(train_rating_values)
        self.train_truths = th.FloatTensor(train_rating_values).to(device)

        self.valid_enc_graph = self.train_enc_graph
        self.valid_dec_graph = self._generate_dec_graph(valid_rating_pairs, review_feat=self.train_review_feat)
        self.valid_labels = _make_labels(valid_rating_values)
        self.valid_truths = th.FloatTensor(valid_rating_values).to(device)

        # self.test_enc_graph = self._generate_enc_graph(all_train_rating_pairs, all_train_rating_values, add_support=True)
        self.test_enc_graph = self.train_enc_graph
        self.test_dec_graph = self._generate_dec_graph(test_rating_pairs, review_feat=self.train_review_feat)
        self.test_labels = _make_labels(test_rating_values)
        self.test_truths = th.FloatTensor(test_rating_values).to(device)

        def _npairs(graph):
            rst = 0
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                rst += graph.number_of_edges(str(r))
            return rst

        print("Train enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.train_enc_graph.number_of_nodes('user'),
            self.train_enc_graph.number_of_nodes('movie'),
            _npairs(self.train_enc_graph)))
        print("Train dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.train_dec_graph.number_of_nodes('user'),
            self.train_dec_graph.number_of_nodes('movie'),
            self.train_dec_graph.number_of_edges()))
        print("Valid enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.valid_enc_graph.number_of_nodes('user'),
            self.valid_enc_graph.number_of_nodes('movie'),
            _npairs(self.valid_enc_graph)))
        print("Valid dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.valid_dec_graph.number_of_nodes('user'),
            self.valid_dec_graph.number_of_nodes('movie'),
            self.valid_dec_graph.number_of_edges()))
        print("Test enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.test_enc_graph.number_of_nodes('user'),
            self.test_enc_graph.number_of_nodes('movie'),
            _npairs(self.test_enc_graph)))
        print("Test dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.test_dec_graph.number_of_nodes('user'),
            self.test_dec_graph.number_of_nodes('movie'),
            self.test_dec_graph.number_of_edges()))

    # def _process_user_item_review_feat(self):
    #     user_list = [[] for _ in range(self._num_user)]
    #     movie_list = [[] for _ in range(self._num_movie)]
    #     for k, v in self.train_review_feat.items():
    #         u, m = k
    #         user_list[u].append(v)
    #         movie_list[m].append(v)
    #     user_list = [torch.stack(x).mean(0) for x in user_list]
    #     movie_list = [torch.stack(x).mean(0) for x in movie_list]
    #
    #     user_list = torch.stack(user_list).to(torch.float32)
    #     movie_list = torch.stack(movie_list).to(torch.float32)
    #     user_list.requires_grad = True
    #     movie_list.requires_grad = True
    #     return user_list, movie_list

    def _process_user_item_review_feat_groupby_rating(self):

        user_id = self.train_datas[0]
        item_id = self.train_datas[1]
        rating = self.train_datas[2]
        ui = list(zip(user_id, item_id))
        ui2r = dict(zip(ui, rating))

        # r -> id -> feature_list
        user_list = [[[] for _ in range(self._num_user)] for r in self.possible_rating_values]
        movie_list = [[[] for _ in range(self._num_movie)] for r in self.possible_rating_values]

        for u, m in zip(user_id, item_id):
        # for k, v in self.train_review_feat.items():
            r = ui2r[(u, m)]
            user_list[r-1][u].append(self.train_review_feat[(u, m)])
            movie_list[r-1][m].append(self.train_review_feat[(u, m)])

        def stack(vector_list):
            if len(vector_list) > 0:
                return torch.stack(vector_list).mean(0)
            else:
                return torch.randn(self._review_fea_size)

        user_list = [[stack(x) for x in r] for r in user_list]
        movie_list = [[stack(x) for x in r] for r in movie_list]

        user_list = [torch.stack(x).to(torch.float32) for x in user_list]
        movie_list = [torch.stack(x).to(torch.float32) for x in movie_list]
        user_review_feat_groupby_rating = {k+1: v for k, v in enumerate(user_list)}
        movie_review_feat_groupby_rating = {k+1: v for k, v in enumerate(movie_list)}
        return user_review_feat_groupby_rating, movie_review_feat_groupby_rating

    def _generate_pair_value(self, sub_dataset):
        """
        :param sub_dataset: all, train, valid, test
        :return:
        """
        if sub_dataset == 'all_train':
            user_id = self.train_datas[0] + self.valid_datas[0]
            item_id = self.train_datas[1] + self.valid_datas[1]
            rating = self.train_datas[2] + self.valid_datas[2]
        elif sub_dataset == 'train':
            user_id = self.train_datas[0]
            item_id = self.train_datas[1]
            rating = self.train_datas[2]
        elif sub_dataset == 'valid':
            user_id = self.valid_datas[0]
            item_id = self.valid_datas[1]
            rating = self.valid_datas[2]
        else:
            user_id = self.test_datas[0]
            item_id = self.test_datas[1]
            rating = self.test_datas[2]

        rating_pairs = (np.array(user_id, dtype=np.int64),
                        np.array(item_id, dtype=np.int64))
        rating_values = np.array(rating, dtype=np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values,
                            add_support=False):
        user_movie_r = np.zeros((self._num_user, self._num_movie),
                                dtype=np.float32)
        user_movie_r[rating_pairs] = rating_values
        record_size = rating_pairs[0].shape[0]
        review_feat_list = [self.train_review_feat[(rating_pairs[0][x], rating_pairs[1][x])] for x in range(record_size)]
        review_feat_list = torch.stack(review_feat_list).to(torch.float32)

        data_dict = dict()
        num_nodes_dict = {'user': self._num_user, 'movie': self._num_movie}
        rating_row, rating_col = rating_pairs
        review_data_dict = dict()
        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('user', str(rating), 'movie'): (rrow, rcol),
                ('movie', 'rev-%s' % str(rating), 'user'): (rcol, rrow)
            })
            review_data_dict[str(rating)] = review_feat_list[ridx]

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        for rating in self.possible_rating_values:
            graph[str(rating)].edata['review_feat'] = review_data_dict[str(rating)]
            graph['rev-%s' % str(rating)].edata['review_feat'] = review_data_dict[str(rating)]

        # sanity check
        assert len(rating_pairs[0]) == sum(
            [graph.number_of_edges(et) for et in graph.etypes]) // 2

        if self.user_doc is not None:
            graph.nodes['user'].data['doc'] = self.user_doc
            graph.nodes['movie'].data['doc'] = self.movie_doc

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)

            user_ci = []
            user_cj = []
            movie_ci = []
            movie_cj = []
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                user_ci.append(graph['rev-%s' % r].in_degrees())
                movie_ci.append(graph[r].in_degrees())
                if self._symm:
                    user_cj.append(graph[r].out_degrees())
                    movie_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    user_cj.append(th.zeros((self.num_user,)))
                    movie_cj.append(th.zeros((self.num_movie,)))
            user_ci = _calc_norm(sum(user_ci))
            movie_ci = _calc_norm(sum(movie_ci))
            if self._symm:
                user_cj = _calc_norm(sum(user_cj))
                movie_cj = _calc_norm(sum(movie_cj))
            else:
                user_cj = th.ones(self.num_user, )
                movie_cj = th.ones(self.num_movie, )
            graph.nodes['user'].data.update({'ci': user_ci, 'cj': user_cj})
            graph.nodes['movie'].data.update({'ci': movie_ci, 'cj': movie_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs, review_feat=None):
        ones = np.ones_like(rating_pairs[0])
        user_movie_ratings_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_user, self.num_movie), dtype=np.float32)
        g = dgl.bipartite_from_scipy(user_movie_ratings_coo, utype='_U',
                                     etype='_E', vtype='_V')
        g = dgl.heterograph({('user', 'rate', 'movie'): g.edges()},
                            num_nodes_dict={'user': self.num_user,
                                            'movie': self.num_movie})
        # e_fea = th.Tensor(rating_pairs).T
        # g.edata['review_feat'] = e_fea
        # g.edata['u_id'] = torch.IntTensor(rating_pairs[0].tolist())
        # g.edata['i_id'] = torch.IntTensor(rating_pairs[1].tolist())
        if review_feat is not None:
            ui = list(zip(rating_pairs[0].tolist(), rating_pairs[1].tolist()))
            feat = [review_feat[x] for x in ui]
            feat = torch.stack(feat, dim=0).float()
            g.edata['review_feat'] = feat

        # if self.user_doc is not None:
        #     g.nodes['user'].data['doc'] = self.user_doc
        #     g.nodes['movie'].data['doc'] = self.movie_doc

        return g

    @property
    def num_links(self):
        return self.possible_rating_values.size

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_movie(self):
        return self._num_movie

    @staticmethod
    def load_ml100k(dataset_path):
        train_path = '/home/d1/shuaijie/data/ml-100k/u1.base'
        test_path = '/home/d1/shuaijie/data/ml-100k/u1.test'
        valid_ratio = 0.1
        train = pd.read_csv(
            train_path, sep='\t', header=None,
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            dtype={'user_id': np.int64, 'item_id': np.int64,
                   'ratings': np.float32, 'timestamp': np.int64},
            engine='python')
        user_size = train['user_id'].max() + 1
        item_size = train['item_id'].max() + 1
        dataset_info = {'user_size': user_size, 'item_size': item_size}
        test = pd.read_csv(
            test_path, sep='\t', header=None,
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            dtype={'user_id': np.int64, 'item_id': np.int64,
                   'ratings': np.float32, 'timestamp': np.int64},
            engine='python')
        num_valid = int(
            np.ceil(train.shape[0] * valid_ratio))
        shuffled_idx = np.random.permutation(train.shape[0])
        valid = train.iloc[shuffled_idx[: num_valid]]
        train = train.iloc[shuffled_idx[num_valid:]]

        return train, valid, test, None, None, dataset_info


def process_doc(doc, word2id, doc_length=256):
    for k, v in doc.items():
        v = [x['review_text'] for x in v]
        v = ' '.join(v)
        v = v.split()[: doc_length]
        v = ' '.join(v)
        v = parse_word_to_idx(word2id, v)
        v = pad_sentence(v, doc_length)
        doc[k] = v
    result = [doc[i] for i in range(len(doc))]
    result = np.stack(result)
    return result


def parse_word_to_idx(word2id, sentence):
    idx = np.array([word2id[x] for x in sentence.split()], dtype=np.int64)
    return idx


def pad_sentence(sentence, length):
    if sentence.shape[0] < length:
        pad_length = length - sentence.shape[0]
        sentence = np.pad(sentence, (0, pad_length), 'constant', constant_values=0)
    return sentence


# if __name__ == '__main__':
#     dataset_name = 'Digital_Music_5'
#     dataset_path = '/home/d1/shuaijie/data/Digital_Music_5/Digital_Music_5.json'
#     dataset = MovieLens(dataset_name,
#                         dataset_path,
#                         'cpu',
#                         review_fea_size=64,
#                         symm=True)
#
#     train_eg = dataset.train_enc_graph
#     pass
#     print(dataset.train_enc_graph)
