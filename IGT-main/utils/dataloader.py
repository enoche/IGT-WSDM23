# coding: utf-8
# @Time   : 2021/05/24
# @Author : enoche
# @email  : enoche.chow@gmail.com
"""
Wrap dataset into dataloader
################################################
"""
import math
import torch
import random
import numpy as np
from logging import getLogger
from scipy.sparse import coo_matrix
import pandas as pd
from utils.utils import lat_lon_distance, get_data_preprocessing
from models.common.general_model import AbstractDataPreprocessing


class AbstractDataLoader(object):
    """:class:`AbstractDataLoader` is an abstract object which would return a batch of data
    And it is also the ancestor of all other dataloader.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        dataset (Dataset): The dataset of this dataloader.
        shuffle (bool): If ``True``, dataloader will shuffle before every epoch.
        real_time (bool): If ``True``, dataloader will do data pre-processing,
            such as neg-sampling and data-augmentation.
        pr (int): Pointer of dataloader.
        step (int): The increment of :attr:`pr` for each batch.
        batch_size (int): The max interaction number for all batch.
    """
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, shuffle=False, training_mode=True):
        self.config = config
        self.logger = getLogger()
        self.dataset = dataset
        self.additional_dataset = additional_dataset
        self.batch_size = batch_size
        self.step = batch_size
        self.shuffle = shuffle
        self.device = config['device']
        self.is_graph_model = config['is_graph_model']
        self.graph_info_dict = {}
        self.training_mode = training_mode

        self.dataset_bk = self.dataset.copy(self.dataset.df)
        self.feature_columns = self.dataset.get_columns()
        self.pr = 0
        self.feature_len = 0
        self.data_preprocessing = self.get_preprocessing()
        if self.data_preprocessing is None:
            self.data_preprocessing = AbstractDataPreprocessing()
        else:
            self.data_preprocessing = self.data_preprocessing()
        self.setup()

    def setup(self):
        raise NotImplementedError('Method [pr_end] should be implemented')

    def get_preprocessing(self):
        return get_data_preprocessing(self.config['model'])

    def reset_loader(self):
        if self.shuffle:
            self.dataset = self.dataset_bk.copy(self.dataset_bk.df)

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            raise StopIteration()
        return self._next_batch_data()

    @property
    def pr_end(self):
        """This property marks the end of dataloader.pr which is used in :meth:`__next__()`."""
        raise NotImplementedError('Method [pr_end] should be implemented')

    def _shuffle(self):
        """Shuffle the order of data, and it will be called by :meth:`__iter__()` if self.shuffle is True.
        """
        raise NotImplementedError('Method [shuffle] should be implemented.')

    def _next_batch_data(self):
        """Assemble next batch of data in form of Interaction, and return these data.

        Returns:
            Interaction: The next batch of data.
        """
        raise NotImplementedError('Method [next_batch_data] should be implemented.')


######################################
####  Normal dataloader
#####################################
class DataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, shuffle=False, training_mode=True):
        super().__init__(config, dataset, additional_dataset=additional_dataset,
                         batch_size=batch_size, shuffle=shuffle, training_mode=training_mode)

    def setup(self):
        """This function is used to do some personalized data preprocess."""
        # for our model, graph based
        if self.config['is_graph_model']:
            assert self.additional_dataset is not None
            if self.training_mode:
                self.adj_matrices = self.graph_adjs(self.dataset.df)
                self.additional_adj_matrices = self.graph_adjs(self.additional_dataset.df)
                self.get_node_features(self.dataset.df)

        self.dataset_bk = self.dataset.copy(self.dataset.df)
        self.feature_columns = self.dataset.get_columns()
        self.feature_len = len(self.feature_columns) - 1
        self.graph_info_dict = self.dataset.graph_info_dict  # number of unique elements

    def graph_adjs(self, df):
        graph_info_dict = self.dataset.graph_info_dict  # number of unique elements
        # build adj matrix: seller_id-->accept_county-->receiver_county-->pay_hour
        sa_matrix = self.get_adj_matrix(df, ['seller_id', 'accept_city'], graph_info_dict)
        ar_matrix = self.get_adj_matrix(df, ['accept_city', 'receiver_county'], graph_info_dict)
        rp_matrix = self.get_adj_matrix(df, ['receiver_county', 'pay_hour'], graph_info_dict)
        return [sa_matrix, ar_matrix, rp_matrix]

    def homo_graph_adjs(self, df, add_df=None):
        temp_df = df if add_df is None else add_df
        min_acc = max(temp_df.seller_id) + 1
        min_rec = min_acc + max(temp_df.accept_city) + 1
        min_hour = min_rec + max(temp_df.receiver_county) + 1
        # reindex
        df['accept_city'] += min_acc
        df['receiver_county'] += min_rec
        df['pay_hour'] += min_hour
        # build adj matrix based on training df to avoid data leakage
        num_nodes = min_hour + 24
        self.num_nodes = num_nodes
        # get indices and counts
        cols = [['seller_id', 'accept_city'], ['accept_city', 'receiver_county'], ['receiver_county', 'pay_hour']]
        values = []
        for col in cols:
            df_group = df.groupby(by=col)
            idx_v = torch.tensor([list(key) + [len(value)] for key, value in df_group.indices.items()])
            values.append(idx_v)
        adj_values = torch.vstack(values).T
        return torch.sparse.FloatTensor(adj_values[:2], adj_values[2], torch.Size([num_nodes, num_nodes]))

    def get_node_features(self, df):
        graph_info_dict = self.dataset.graph_info_dict  # number of unique elements
        self.fea_dim_seller = 57
        self.fea_dim_accept = 121
        self.fea_dim_receiver = 85
        seller_fea = np.zeros((graph_info_dict['seller_id'], self.fea_dim_seller))
        accept_city_fea = np.zeros((graph_info_dict['accept_city'], self.fea_dim_accept))
        rcv_county_fea = np.zeros((graph_info_dict['receiver_county'], self.fea_dim_receiver))
        self.node_features = {'seller_id': seller_fea, 'accept_city': accept_city_fea,
                              'receiver_county': rcv_county_fea}

        # init feature for seller_id, accept_county, receiver_county, pay_hour: averaged logs
        fea_dim_list = [self.fea_dim_seller, self.fea_dim_accept, self.fea_dim_receiver]
        self.get_features(df, 'seller_id', 0, fea_dim_list[0])
        self.get_features(df, 'accept_city', fea_dim_list[0], fea_dim_list[0] + fea_dim_list[1])
        self.get_features(df, 'receiver_county', fea_dim_list[0] + fea_dim_list[1], sum(fea_dim_list))

    def get_rela_node_features(self, df):
        graph_info_dict = self.dataset.graph_info_dict  # number of unique elements
        self.fea_dim_seller = 57
        self.fea_dim_accept = 121
        self.fea_dim_receiver = 85
        max_len = max(self.fea_dim_seller, self.fea_dim_accept, self.fea_dim_receiver)
        self.homo_node_features = np.zeros((self.num_nodes, max_len))
        # init feature
        fea_dim_list = [self.fea_dim_seller, self.fea_dim_accept, self.fea_dim_receiver]
        self.get_rela_features(df, 'seller_id', 0, fea_dim_list[0], max_len)
        self.get_rela_features(df, 'accept_city', fea_dim_list[0], fea_dim_list[0] + fea_dim_list[1], max_len)
        self.get_rela_features(df, 'receiver_county', fea_dim_list[0] + fea_dim_list[1], sum(fea_dim_list), max_len)

    def graph_preprocessing(self, df):
        graph_info_dict = self.dataset.graph_info_dict  # number of unique elements
        self.fea_dim_seller = 57
        self.fea_dim_accept = 121
        self.fea_dim_receiver = 85
        seller_fea = np.zeros((graph_info_dict['seller_id'], self.fea_dim_seller))
        accept_city_fea = np.zeros((graph_info_dict['accept_city'], self.fea_dim_accept))
        rcv_county_fea = np.zeros((graph_info_dict['receiver_county'], self.fea_dim_receiver))
        self.node_features = {'seller_id': seller_fea, 'accept_city': accept_city_fea,
                              'receiver_county': rcv_county_fea}

        # build adj matrix: seller_id-->accept_county-->receiver_county-->pay_hour
        sa_matrix = self.get_adj_matrix(df, ['seller_id', 'accept_city'], graph_info_dict)
        ar_matrix = self.get_adj_matrix(df, ['accept_city', 'receiver_county'], graph_info_dict)
        rp_matrix = self.get_adj_matrix(df, ['receiver_county', 'pay_hour'], graph_info_dict)
        self.adj_matrices = [sa_matrix, ar_matrix, rp_matrix]

        # init feature for seller_id, accept_county, receiver_county, pay_hour: averaged logs
        fea_dim_list = [self.fea_dim_seller, self.fea_dim_accept, self.fea_dim_receiver]
        self.get_features(df, 'seller_id', 0, fea_dim_list[0])
        self.get_features(df, 'accept_city', fea_dim_list[0], fea_dim_list[0] + fea_dim_list[1])
        self.get_features(df, 'receiver_county', fea_dim_list[0] + fea_dim_list[1], sum(fea_dim_list))

    def get_rela_features(self, df, col, start, end, max_len):
        df_key = df.groupby([col]).mean()
        idx = df_key.index.values.astype(np.int_)
        v = df_key.values.astype(np.float32)[:, start:end]
        v = np.pad(v, [(0, 0), (0, max_len - end + start)], mode='constant', constant_values=0)
        self.homo_node_features[idx] = v

    # average the features across all time-lines
    def get_features(self, df, col, start, end):
        df_key = df.groupby([col]).mean()
        idx = df_key.index.values.astype(np.int_)
        v = df_key.values.astype(np.float32)[:, start:end]
        fea_arr = self.node_features[col]
        fea_arr[idx] = v

    def get_adj_matrix(self, df, cols, g_dict):
        assert len(cols) == 2
        df_group = df.groupby(by=cols)
        # indices and counts
        idx_v = torch.tensor([list(key) + [len(value)] for key, value in df_group.indices.items()]).t()
        return torch.sparse.FloatTensor(idx_v[:2], idx_v[2], torch.Size([g_dict[cols[0]], g_dict[cols[1]]]))

    @property
    def pr_end(self):
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()

    # Models like LR, directly get full data, no batch
    def full_data(self):
        if self.shuffle:
            self._shuffle()
        # no node ids, date
        data_X = self.dataset[:, :-6].values
        data_y = self.dataset[:, -1].values
        return data_X, data_y

    def _next_batch_data(self):
        cur_data = self.dataset[self.pr: self.pr + self.step]
        self.pr += self.step
        data_X = cur_data.iloc[:, :-1].values
        data_y = cur_data.iloc[:, -1].values
        return data_X, data_y
