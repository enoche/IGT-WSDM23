# coding: utf-8
# @Time   : 2021/05/24
# @Author : enoche
# @Email  : enoche.chow@gmail.com
#
"""
##########################
"""

import numpy as np
import torch
import torch.nn as nn


class GeneralModel(nn.Module):
    r"""Base class for all models
    """
    def __init__(self, config, dataloader):
        super(GeneralModel, self).__init__()
        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']
        self.dataloader = dataloader

    def fit(self, epoch):
        raise NotImplementedError

    def predict(self, orders):
        r"""Predict the scores between users and items.
        Args:
            interaction (Interaction): Interaction class of the batch.
        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class AbstractDataPreprocessing:
    r"""Base class for all models to preprecessing data
        """
    def __init__(self):
        pass

    def preprocessing(self, config, df):
        """This function is used to do some data preprocess."""
        # full columns
        # ['seller_id', 'express_code', 'accept_city', 'accept_county',
        # 'accept_county_lon', 'accept_county_lat', 'receiver_county', 'receiver_county_lon', 'receiver_county_lat',
        # 'receiver_city', 'receiver_prov',
        # 'pay_hour',
        # 'pay_accept_inter', 'accept_hour', 'accept_ac1_inter', 'ac1_hour', 'ac1_ac2_inter', 'ac2_hour',
        # 'ac2_ac3_inter', 'ac3_hour', 'ac3_deliver_inter', 'deliver_station_hour', 'deliver_out_inter',
        # 'deliver_sign_inter', 'sign_hour', 'accept_fac_id', 'ac1_fac_id', 'ac2_fac_id', 'ac3_fac_id', 'deliver_fac_id',
        # 'date',
        # 'sin_pay_hour', 'cos_pay_hour', 'sin_dt_dayofweek', 'cos_dt_dayofweek', 'sin_dt_day', 'cos_dt_day',
        # 'sin_dt_month', 'cos_dt_month',
        # 'express_  one hot code',
        # 'total_cost']

        # not graph model, only keep features
        drop_columns = ['f1', 'seller_id', 'express_code', 'accept_city', 'accept_county',
                        'receiver_county', 'receiver_city', 'receiver_prov',
                        'pay_accept_inter', 'accept_hour', 'accept_ac1_inter', 'ac1_hour', 'ac1_ac2_inter', 'ac2_hour',
                        'ac2_ac3_inter', 'ac3_hour', 'ac3_deliver_inter', 'deliver_station_hour', 'deliver_out_inter',
                        'deliver_sign_inter', 'sign_hour', 'accept_fac_id', 'ac1_fac_id', 'ac2_fac_id', 'ac3_fac_id',
                        'deliver_fac_id', 'date']
        # keep latitude and longitude, and total_cost
        df.drop(drop_columns, inplace=True, axis=1)
        return df


