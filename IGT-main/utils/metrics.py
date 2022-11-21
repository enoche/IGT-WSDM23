# -*- encoding: utf-8 -*-
# @Time    :   2021/06/02
# @Author  :   enoche
# @email   :   enoche.chow@gmail.com

"""
############################
"""

from logging import getLogger
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def mae_(trues, preds):
    r"""`Mean absolute error regression loss`__

    .. __: https://en.wikipedia.org/wiki/Mean_absolute_error

    .. math::
        \mathrm{MAE}=\frac{1}{|{T}|} \sum_{(u, i) \in {T}}\left|\hat{r}_{u i}-r_{u i}\right|

    :math:`T` is the test set, :math:`\hat{r}_{u i}` is the score predicted by the model,
    and :math:`r_{u i}` the actual score of the test set.

    """
    return mean_absolute_error(trues, preds)

def rmse_(trues, preds):
    r"""`Mean std error regression loss`__

    .. __: https://en.wikipedia.org/wiki/Root-mean-square_deviation

    .. math::
        \mathrm{RMSE} = \sqrt{\frac{1}{|{T}|} \sum_{(u, i) \in {T}}(\hat{r}_{u i}-r_{u i})^{2}}

    :math:`T` is the test set, :math:`\hat{r}_{u i}` is the score predicted by the model,
    and :math:`r_{u i}` the actual score of the test set.

    """
    return np.sqrt(mean_squared_error(trues, preds))

def mare_(trues, preds):
    return sum(abs(trues-preds))/sum(trues)
    pass

def mape_(trues, preds):
    return mean_absolute_percentage_error(trues, preds)

"""Function name and function mapper.
Useful when we have to serialize evaluation metric names
and call the functions based on deserialized names
"""
metrics_dict = {
    'mae': mae_,
    'rmse': rmse_,
    'mare': mare_,
    'mape': mape_
}
