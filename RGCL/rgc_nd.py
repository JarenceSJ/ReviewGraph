# -*- coding: utf-8 -*-

import argparse
import math
import random
import string
from abc import ABC

import torch as th
from torch.nn import init

from data import MovieLens
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from util import *


def config():
    parser = argparse.ArgumentParser(description='RGC-ND')
    parser.add_argument('--device', default='0', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('-dn', '--dataset_name', type=str,
                        help='dataset name')
    parser.add_argument('-dp', '--dataset_path', type=str,
                        help='raw dataset file path')
    parser.add_argument('--model_save_path', type=str,
                        help='The model saving path')
    parser.add_argument('--model_activation', type=str, default="leaky")
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")

    parser.add_argument('--gcn_dropout', type=float, default=0.7)
    parser.add_argument('--train_max_iter', type=int, default=2000)
    parser.add_argument('--train_log_interval', type=int, default=1)
    parser.add_argument('--train_valid_interval', type=int, default=1)
    parser.add_argument('--train_optimizer', type=str, default="Adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_min_lr', type=float, default=0.001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_decay_patience', type=int, default=50)
    parser.add_argument('--train_early_stopping_patience', type=int, default=100)
    parser.add_argument('--share_param', default=False, action='store_true')
    parser.add_argument('--train_classification', type=bool, default=True)
    parser.add_argument('--review_feat_size', type=int, default=64)

    parser.add_argument('--nd_alpha', type=float, default=1.)

    args = parser.parse_args()
    args.model_short_name = 'RGC-ND'

    args.dataset_name = 'Digital_Music_5'
    args.dataset_path = '/home/d1/shuaijie/data/Digital_Music_5/Digital_Music_5.json'
    args.review_feat_size = 64
    args.gcn_dropout = 0.70  # 0.7
    args.nd_alpha = 1.0  # 1.0
    args.device = 1
    args.train_max_iter = 400

    args.device = th.device(args.device) if args.device >= 0 else th.device('cpu')

    # configure save_fir to save all the info
    if args.model_save_path is None:
        args.model_save_path = 'log/' \
                               + args.model_short_name \
                               + '_' + args.dataset_name \
                               + '_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=2)) \
                               + '.pkl'
    if not os.path.isdir('log'):
        os.makedirs('log')

    args.gcn_agg_units = args.review_feat_size
    args.gcn_out_units = args.review_feat_size

    return args


class GCMCGraphConv(nn.Module, ABC):

    def __init__(self,
                 in_feats,
                 out_feats,
                 device=None,
                 dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))

        self.prob_score = nn.Linear(out_feats, 1, bias=False)
        self.review_score = nn.Linear(out_feats, 1, bias=False)
        self.review_w = nn.Linear(out_feats, out_feats, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.prob_score.weight)
        init.xavier_uniform_(self.review_score.weight)
        init.xavier_uniform_(self.review_w.weight)

    def forward(self, graph, feat, weight=None):

        with graph.local_scope():

            feat = self.weight

            graph.srcdata['h'] = feat
            review_feat = graph.edata['review_feat']

            graph.edata['pa'] = torch.sigmoid(self.prob_score(review_feat))
            graph.edata['ra'] = torch.sigmoid(self.review_score(review_feat))

            graph.edata['rf'] = self.review_w(review_feat)
            graph.update_all(lambda edges: {'m': (edges.src['h'] * edges.data['pa']
                                                  + edges.data['rf'] * edges.data['ra'])
                                                 * self.dropout(edges.src['cj'])},
                             fn.sum(msg='m', out='h'))

            rst = graph.dstdata['h']
            rst = rst * graph.dstdata['ci']

        return rst


class GCMCLayer(nn.Module, ABC):

    def __init__(self,
                 rating_vals,
                 user_in_units,
                 movie_in_units,
                 out_units,
                 dropout_rate=0.0,
                 device=None):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.ufc = nn.Linear(out_units, out_units)
        self.ifc = nn.Linear(out_units, out_units)
        self.dropout = nn.Dropout(dropout_rate)
        sub_conv = {}
        for rating in rating_vals:
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            sub_conv[rating] = GCMCGraphConv(user_in_units,
                                             out_units,
                                             device=device,
                                             dropout_rate=dropout_rate)
            sub_conv[rev_rating] = GCMCGraphConv(movie_in_units,
                                                 out_units,
                                                 device=device,
                                                 dropout_rate=dropout_rate)
        self.conv = dglnn.HeteroGraphConv(sub_conv, aggregate='sum')
        self.agg_act = nn.GELU()
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, ufeat=None, ifeat=None):
        in_feats = {'user': ufeat, 'movie': ifeat}
        out_feats = self.conv(graph, in_feats)
        ufeat = out_feats['user']
        ifeat = out_feats['movie']
        ufeat = ufeat.view(ufeat.shape[0], -1)
        ifeat = ifeat.view(ifeat.shape[0], -1)

        # fc and non-linear
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)
        return ufeat, ifeat


class MLPPredictor(nn.Module, ABC):

    def __init__(self, in_units, num_classes):

        super(MLPPredictor, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_units * 2, in_units, bias=False),
            nn.GELU(),
            nn.Linear(in_units, in_units, bias=False),
        )
        self.predictor = nn.Linear(in_units, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        h_fea = self.linear(th.cat([h_u, h_v], dim=1))
        score = self.predictor(h_fea).squeeze()
        return {'score': score}

    def forward(self, graph, ufeat, ifeat):
        graph.nodes['movie'].data['h'] = ifeat
        graph.nodes['user'].data['h'] = ufeat
        with graph.local_scope():
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class ContrastLoss(nn.Module, ABC):

    def __init__(self, feat_size):
        super(ContrastLoss, self).__init__()
        self.bilinear = nn.Bilinear(feat_size, feat_size, 1)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x, y):
        """
        :param x: bs * dim
        :param y: bs * dim
        :param y_neg: bs * k * dim
        :return:
        """

        # # positive
        scores = self.bilinear(x, y).squeeze()
        labels = th.ones(scores.shape).to(scores.device)

        pos_loss = self.bce_loss(scores, labels)

        idx = th.randperm(y.shape[0])
        y_neg = y[idx, :]

        neg2_scores = self.bilinear(x, y_neg).squeeze()
        neg2_labels = th.zeros(neg2_scores.shape).to(scores.device)

        neg2_loss = self.bce_loss(neg2_scores, neg2_labels)

        loss = pos_loss + neg2_loss

        return loss


class Net(nn.Module, ABC):
    def __init__(self, params):
        super(Net, self).__init__()
        self._params = params
        self._act = get_activation(params.model_activation)
        self.encoder = GCMCLayer(params.rating_vals,
                                 params.src_in_units,
                                 params.dst_in_units,
                                 params.gcn_out_units,
                                 dropout_rate=params.gcn_dropout,
                                 device=params.device)

        self.contrast_loss = ContrastLoss(params.gcn_out_units)

        if params.train_classification:
            self.decoder = MLPPredictor(in_units=params.gcn_out_units,
                                        num_classes=len(params.rating_vals))
        else: 
            self.decoder = MLPPredictor(in_units=params.gcn_out_units,
                                        num_classes=1)

    def forward(self, enc_graph, dec_graph, ufeat, ifeat):
        user_out, movie_out = self.encoder(enc_graph, ufeat, ifeat)
        pred_ratings = self.decoder(dec_graph, user_out, movie_out)
        return pred_ratings, user_out, movie_out


def evaluate(params, net, dataset, segment='valid'):
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = th.FloatTensor(possible_rating_values).to(params.device)

    if segment == "valid":
        rating_values = dataset.valid_truths
        enc_graph = dataset.valid_enc_graph
        dec_graph = dataset.valid_dec_graph
    elif segment == "test":
        rating_values = dataset.test_truths
        enc_graph = dataset.test_enc_graph
        dec_graph = dataset.test_dec_graph
    else:
        raise NotImplementedError

    # Evaluate RMSE
    net.eval()
    with th.no_grad():
        pred_ratings, _, _ = net(enc_graph, dec_graph, dataset.user_feature, dataset.movie_feature)
        if params.train_classification:
            real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                                 nd_possible_rating_values.view(1, -1)).sum(dim=1)
            rmse = ((real_pred_ratings - rating_values) ** 2.).mean().item()
        else:
            rmse = ((pred_ratings - rating_values) ** 2.).mean().item()
        rmse = np.sqrt(rmse)
        # real_pred_ratings = (th.softmax(pred_ratings, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
        # rmse = ((real_pred_ratings - rating_values) ** 2.).mean().sqrt().item()
    return rmse


def train(params):

    print(params)
    dataset = MovieLens(params.dataset_name,
                        params.dataset_path,
                        params.device,
                        params.review_feat_size,
                        symm=params.gcn_agg_norm_symm)
    print("Loading data finished ...\n")

    params.src_in_units = dataset.user_feature_shape[1]
    params.dst_in_units = dataset.movie_feature_shape[1]
    params.rating_vals = dataset.possible_rating_values

    # build the net
    net = Net(params=params)
    net = net.to(params.device)
    nd_possible_rating_values = th.FloatTensor(dataset.possible_rating_values).to(params.device)
    rating_loss_net = nn.CrossEntropyLoss() if params.train_classification else nn.MSELoss()
    learning_rate = params.train_lr
    optimizer = get_optimizer(params.train_optimizer)(net.parameters(), lr=learning_rate)
    print("Loading network finished ...\n")

    # prepare training data
    if params.train_classification:
        train_gt_labels = dataset.train_labels
        train_gt_ratings = dataset.train_truths
    else:
        train_gt_labels = dataset.train_truths.float()
        train_gt_ratings = dataset.train_truths.float()

    # declare the loss information
    best_valid_rmse = np.inf
    best_test_rmse = np.inf
    no_better_valid = 0
    best_iter = -1

    dataset.train_enc_graph = dataset.train_enc_graph.int().to(params.device)
    dataset.train_dec_graph = dataset.train_dec_graph.int().to(params.device)
    dataset.valid_enc_graph = dataset.train_enc_graph
    dataset.valid_dec_graph = dataset.valid_dec_graph.int().to(params.device)
    dataset.test_enc_graph = dataset.test_enc_graph.int().to(params.device)
    dataset.test_dec_graph = dataset.test_dec_graph.int().to(params.device)

    print("Start training ...")
    for iter_idx in range(1, params.train_max_iter):
        net.train()
        pred_ratings1, user1, item1 = net(dataset.train_enc_graph,
                                          dataset.train_dec_graph,
                                          dataset.user_feature,
                                          dataset.movie_feature)
        pred_ratings2, user2, item2 = net(dataset.train_enc_graph,
                                          dataset.train_dec_graph,
                                          dataset.user_feature,
                                          dataset.movie_feature)
        loss1 = rating_loss_net(pred_ratings1, train_gt_labels).mean()
        loss2 = rating_loss_net(pred_ratings2, train_gt_labels).mean()
        r_loss = (loss1 + loss2) / 2

        user_mi_loss = net.contrast_loss(user1, user2)
        item_mi_loss = net.contrast_loss(item1, item2)
        nd_loss = (user_mi_loss + item_mi_loss) / 2

        total_loss = r_loss + params.nd_alpha * nd_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), params.train_grad_clip)
        optimizer.step()

        if params.train_classification:
            real_pred_ratings = (th.softmax(pred_ratings1, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
        else: 
            real_pred_ratings = pred_ratings1

        train_rmse = ((real_pred_ratings - train_gt_ratings) ** 2).mean().sqrt()

        valid_rmse = evaluate(params=params, net=net, dataset=dataset, segment='valid')
        logging_str = f"Iter={iter_idx:>4d}, Train_RMSE={train_rmse:.4f}, ND_MI={nd_loss:.4f}, " \
                      f"Valid_RMSE={valid_rmse:.4f}, "

        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            no_better_valid = 0
            best_iter = iter_idx
            test_rmse = evaluate(params=params, net=net, dataset=dataset, segment='test')
            best_test_rmse = test_rmse

            logging_str += ' Test RMSE={:.4f}'.format(test_rmse)

        else:
            no_better_valid += 1
            if no_better_valid > params.train_early_stopping_patience and learning_rate <= params.train_min_lr:
                print("Early stopping threshold reached. Stop training.")
                break
            if no_better_valid > params.train_decay_patience:
                new_lr = max(learning_rate * params.train_lr_decay_factor, params.train_min_lr)
                if new_lr < learning_rate:
                    learning_rate = new_lr
                    print("\tChange the LR to %g" % new_lr)
                    for p in optimizer.param_groups:
                        p['lr'] = learning_rate
                    no_better_valid = 0

        print(logging_str)
    print(f'Best Iter Idx={best_iter:>4d}, Best Valid RMSE={best_valid_rmse:.4f}, Best Test RMSE={best_test_rmse:.4f}')
    print(params.model_save_path)


if __name__ == '__main__':
    config_args = config()

    train(config_args)

