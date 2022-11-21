# coding: utf-8
# @Time   : 2021/05/24
# @Author : enoche
# @Email  : enoche.chow@gmail.com

"""
Data pre-processing
##########################
"""
from logging import getLogger
from collections import Counter
import os, copy, csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import reduce
from scipy.stats import zscore


class ETADataset(object):
    def __init__(self, config, df=None, graph_info_dict=None):
        self.config = config
        self.dataset_name = config['dataset']
        self.preprocessed_dataset_path = os.path.abspath(config['preprocessed_data'] + self.dataset_name + '/')
        self.logger = getLogger()

        if df is not None:
            self.df = df
            # from copy, sub-graph
            self.graph_info_dict = graph_info_dict
            return
        # load from preprocessed path?
        if self._load_preprocessed_dataset():
            self.logger.info('\nData loaded from preprocessed dir: ' + self.preprocessed_dataset_path + '\n')
        else:
            raise ValueError('File {} not exist'.format(self.preprocessed_dataset_path))
        # get my unique counts
        self.df = self.df.sort_values(by=['date'])
        self.graph_info_dict = dict(zip([i for i in self.df.columns],
                                       [len(self.df[i].unique()) for i in self.df.columns]))

    def _load_preprocessed_dataset(self):
        file_path = os.path.join(self.preprocessed_dataset_path, '{}_processed.csv'.format(self.dataset_name))
        if not os.path.isfile(file_path):
            return False
        # load
        self.df = self._load_df_from_file(file_path, have_header=True)
        return True

    def _load_df_from_file(self, file_path, have_header=None):
        if have_header:
            df = pd.read_csv(file_path, sep=self.config['field_separator'])
        else:
            df = pd.read_csv(file_path, sep=self.config['field_separator'], header=None)
        return df

    #################################################
    def split(self, days):
        """Split the orders by validation, test days.
        :param days: [validation, test] days
        """
        dt_id = 'date'
        dates = np.sort(self.df[dt_id].unique())
        if sum(days) >= len(dates):
            raise ValueError('No training days left in dataset, total: {}, [valid, test]: {}'.format(len(dates), days))

        # encode time/date, pay_hour & date
        self.df['sin_pay_hour'] = np.sin(2 * np.pi * self.df['pay_hour'] / 24.0)
        self.df['cos_pay_hour'] = np.cos(2 * np.pi * self.df['pay_hour'] / 24.0)

        temp_dt = pd.to_datetime(self.df['date'], format='%Y%m%d')
        # dayofweek start from 0, others from 1
        self.df['sin_dt_dayofweek'] = np.sin(2 * np.pi * temp_dt.dt.dayofweek / 7.0)
        self.df['cos_dt_dayofweek'] = np.cos(2 * np.pi * temp_dt.dt.dayofweek / 7.0)
        self.df['sin_dt_day'] = np.sin(2 * np.pi * temp_dt.dt.day / temp_dt.dt.daysinmonth)
        self.df['cos_dt_day'] = np.cos(2 * np.pi * temp_dt.dt.day / temp_dt.dt.daysinmonth)
        self.df['sin_dt_month'] = np.sin(2 * np.pi * temp_dt.dt.month / 12.0)
        self.df['cos_dt_month'] = np.cos(2 * np.pi * temp_dt.dt.month / 12.0)

        # seller/sender info/receiver info/time&express info
        new_cols = self.df.columns.to_list()
        new_cols.remove('total_cost')
        new_cols.append('total_cost')  # append last
        self.df = self.df[new_cols]

        # get splitting dates
        test_date = dates[-days[1]]
        valid_date = dates[-(days[0]+days[1])]

        # split df based on global time
        dfs = []
        start = dates[0]
        for i in [valid_date, test_date]:
            dfs.append(self.df.loc[(start <= self.df[dt_id]) & (self.df[dt_id] < i)].copy())
            start = i
        # last
        dfs.append(self.df.loc[start <= self.df[dt_id]].copy())
        # filter out new users in valid/test dataset? TODO
        for _df in dfs:
            _df.reset_index(drop=True, inplace=True)
        # wrap as RecDataset
        full_ds = [self.copy(_) for _ in dfs]
        return full_ds

    def copy(self, new_df):
        """Given a new interaction feature, return a new :class:`Dataset` object,
                whose interaction feature is updated with ``new_df``, and all the other attributes the same.

                Args:
                    new_df (pandas.DataFrame): The new interaction feature need to be updated.

                Returns:
                    :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
                """
        nxt = ETADataset(self.config, new_df, self.graph_info_dict)
        return nxt

    def num(self, field):
        """Given ``field``, for token-like fields, return the number of different tokens after remapping,
        for float-like fields, return ``1``.

        Args:
            field (str): field name to get token number.

        Returns:
            int: The number of different tokens (``1`` if ``field`` is a float-like field).
        """
        if field not in self.df.columns:
            raise ValueError('field [{}] not defined in dataset'.format(field))
        uni_len = len(pd.unique(self.df[field]))
        return uni_len

    def shuffle(self):
        """Shuffle the interaction records inplace.
        """
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def get_columns(self):
        return self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Series result
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()
    #
    # def __str__(self):
    #     info = ['']
    #     inter_num = len(self.df)
    #     uni_dates = len(pd.unique(self.df['date']))
    #     info.extend(['The number of orders: {}'.format(inter_num),
    #                  'Total dates: {}'.format(uni_dates),
    #                  'Average orders per day: {}'.format(inter_num/uni_dates)])
    #
    #     return '\n'.join(info)

    def __str__(self):
        info = ['']
        inter_num = len(self.df)
        uni_seller = len(pd.unique(self.df['seller_id']))
        uni_dates = len(pd.unique(self.df['date']))
        avg_actions_of_users = inter_num/uni_seller
        info.extend(['The number of orders: {}'.format(inter_num),
                     'The number of users: {}'.format(uni_seller),
                     'Average orders of users: {}'.format(avg_actions_of_users),
                     'Total dates: {}'.format(uni_dates),
                     'Average orders per day: {}'.format(inter_num/uni_dates)])

        return '\n'.join(info)
