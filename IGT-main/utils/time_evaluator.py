# -*- encoding: utf-8 -*-
# @Time    :   2020/08/04
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :   2020/08/04, 2020/08/11
# @Author  :   Kaiyuan Li, Yupeng Hou
# @email   :   tsotfsk@outlook.com, houyupeng@ruc.edu.cn


# UPDATE
# @Time    :   2021/08/04
# @Author  :   enoche
# @email   :   enoche.chow@gmail.com


"""
################################
"""
import os
import numpy as np
import pandas as pd
import torch
from utils.metrics import metrics_dict
from torch.nn.utils.rnn import pad_sequence
from utils.utils import get_local_time


# These metrics are typical in topk recommendations
#time_metrics = {metric.lower(): metric for metric in ['MAE']}


class TimeEvaluator(object):
    r"""TopK Evaluator is mainly used in ranking tasks. Now, we support six topk metrics which
    contain `'Hit', 'Recall', 'MRR', 'Precision', 'NDCG', 'MAP'`.

    Note:
        The metrics used calculate group-based metrics which considers the metrics scores averaged
        across users. Some of them are also limited to k.

    """
    def __init__(self, config):
        self.config = config
        self.metrics = config['metrics']
        self.save_pred_result = config['save_results']
        self._check_args()

    def evaluate(self, act_values, pred_values, is_test=False, idx=0):
        """calculate the metrics of all batches. It is called at the end of each epoch

        Args:
            batch_matrix_list (list): the results of all batches
            eval_data (Dataset): the class of test data
            is_test: in testing?

        Returns:
            dict: such as ``{'Hit@20': 0.3824, 'Recall@20': 0.0527, 'Hit@10': 0.3153, 'Recall@10': 0.0329}``

        """
        # if save recommendation result?
        if self.save_pred_result and is_test:
            dataset_name = self.config['dataset']
            model_name = self.config['model']
            dir_name = os.path.abspath(self.config['predict_results'])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            file_path = os.path.join(dir_name, '{}-{}-idx{}-{}.csv'.format(
                model_name, dataset_name, idx, get_local_time()))
            act_pre_values = np.vstack((act_values, pred_values))
            x_df = pd.DataFrame(act_pre_values.T)
            x_df.columns = ['true', 'pre']
            x_df.to_csv(file_path, sep='\t', index=False)
        # get metrics
        metric_dict = {}
        result_list = self.metrics_info(act_values, pred_values)
        for metric, value in zip(self.metrics, result_list):
            metric_dict[metric] = round(value, 4)
        return metric_dict

    def _check_args(self):
        # Check metrics
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                self.metrics = [self.metrics]
        else:
            raise TypeError('metrics must be str or list')

        # Convert metric to lowercase
        for m in self.metrics:
            if m.lower() not in metrics_dict:
                raise ValueError("There is no metric named {}!".format(m))
        self.metrics = [metric.lower() for metric in self.metrics]

    def metrics_info(self, act_values, pred_values):
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(act_values, pred_values)
            result_list.append(result)
        return result_list

    def __str__(self):
        mesg = 'The ETA Evaluator Info:\n' + '\tMetrics:[' + ', '.join(
            [metrics_dict[metric.lower()] for metric in self.metrics]) \
               + ']'
        return mesg
