# coding: utf-8
# @Time   : 2021/05/24
# @Author : enoche
# @Email  : enoche.chow@gmail.com
#
"""
Linear-regression
##########################
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from models.common.general_model import GeneralModel


class LR(GeneralModel):
    def __init__(self, config, data):
        super(LR, self).__init__(config, data)
        self.lr = LinearRegression()
        self.loss = mean_absolute_error

    def fit(self, epoch):
        x, y = self.dataloader.full_data()
        self.lr.fit(x, y)
        # predict
        y_pred = self.lr.predict(x)
        return self.loss(y_pred, y)

    def predict(self, orders):
        return self.lr.predict(orders)

