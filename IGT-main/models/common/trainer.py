# coding: utf-8
# @Time   : 2021/05/24
# @Author : enoche
# @Email  : enoche.chow@gmail.com
#
"""
Trainer
##########################
"""


import os
import itertools
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str
from utils.time_evaluator import TimeEvaluator
import torch.optim as optim


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.learner = config['learner'].lower()
        self.learning_rate = config['learning_rate']
        self.optimizer = self.build_optimizer()

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')

    def build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.learner == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.learner == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif self.learner == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            optimizer = None
        return optimizer


class BatchTrainer(AbstractTrainer):
    def __init__(self, config, model):
        super(BatchTrainer, self).__init__(config, model)

        self.logger = getLogger()

        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']

        # save model
        self.checkpoint_dir = config['checkpoint_dir']
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -1 if self.valid_metric_bigger else 1e+10
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.evaluator = TimeEvaluator(config)

    def _train_epoch(self, train_data, loss_func=None):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = 0.0
        for batch_idx, (orders, y) in enumerate(train_data):
            self.optimizer.zero_grad()
            loss = loss_func(orders, y)
            total_loss += loss.item()
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
        return total_loss

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data, load_best_model=False)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['mae']
        return valid_score, valid_result

    def _save_checkpoint(self, epoch):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
        }
        torch.save(state, self.saved_model_file)

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
        resume_file = str(resume_file)
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_valid_score = checkpoint['best_valid_score']

        # load architecture params from checkpoint
        if checkpoint['config']['model'].lower() != self.config['model'].lower():
            self.logger.warning('Architecture configuration given in config file is different from that of checkpoint. '
                                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])

        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data)
            self.train_loss_dict[epoch_idx] = train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \t' + dict2str(valid_result)
                # test
                _, test_result = self._valid_epoch(test_data)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \t' + dict2str(test_result))
                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = '██ Saving current best: %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            is_test: is in testing?
            idx: current hyper-parameter loop index

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        if load_best_model:
            if model_file:      # load from other file
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        batch_pre_arr = np.array([])
        batch_act_arr = np.array([])
        for batch_idx, batched_data in enumerate(eval_data):
            # predict
            scores = self.model.predict(batched_data[0])
            batch_pre_arr = np.append(batch_pre_arr, scores.cpu().numpy() if self.config['use_gpu'] else scores)
            batch_act_arr = np.append(batch_act_arr, batched_data[1])
        # if save results
        if self.config['save_results'] and is_test:
            pass
        return self.evaluator.evaluate(batch_act_arr, batch_pre_arr, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)


class FullTrainer(AbstractTrainer):
    def __init__(self, config, model):
        super(FullTrainer, self).__init__(config, model)

        self.logger = getLogger()

        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']

        # save model
        self.checkpoint_dir = config['checkpoint_dir']
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -1 if self.valid_metric_bigger else 1e+10
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.evaluator = TimeEvaluator(config)

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data, load_best_model=False)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['mae']
        return valid_score, valid_result

    def _save_checkpoint(self, epoch):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'optimizer': None if self.optimizer is None else self.optimizer.state_dict(),
        }
        torch.save(state, self.saved_model_file)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self.model.fit(epoch_idx)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: ' + dict2str(valid_result)
                # test
                #_, test_result = self._valid_epoch(test_data)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    #self.logger.info('test result: \n' + dict2str(test_result))
                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = 'Saving current best: %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result


    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            is_test: is in testing?
            idx: current hyper-parameter loop index

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        if load_best_model:
            if model_file:      # load from other file
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        # batch full users
        batch_pre_arr = np.array([])
        batch_act_arr = np.array([])
        # for batch_idx, batched_data in enumerate(eval_data):
        #     # predict
        #     scores = self.model.predict(batched_data[0])
        #     batch_pre_arr = np.append(batch_pre_arr, scores if type(scores) is np.ndarray else scores.cpu())
        #     batch_act_arr = np.append(batch_act_arr, batched_data[1])
        batched_data = eval_data.full_data()
        scores = self.model.predict(batched_data[0])
        batch_pre_arr = np.append(batch_pre_arr, scores if type(scores) is np.ndarray else scores.cpu())
        batch_act_arr = np.append(batch_act_arr, batched_data[1])
        # temp disable save results for traditional models
        return self.evaluator.evaluate(batch_act_arr, batch_pre_arr, is_test=False, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)

    def plot_feature_importance(self, feature_list, save_path='rslt_plots'):
        if hasattr(self.model, 'get_feature_importances'):
            importances = list(self.model.get_feature_importances())
            feature_importances = [(feature, round(importance, 2)) for feature, importance in
                                   zip(feature_list, importances)]
            # Sort the feature importances by most important first
            feature_importances = sorted(feature_importances, key=lambda x: x[1],
                                         reverse=True)  # Print out the feature and importances
            [self.logger.info('Feature: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
            # plot
            # Set the style
            plt.style.use('fivethirtyeight')  # list of x locations for plotting
            features20 = [pair[0] for pair in feature_importances[:20]]
            importances20 = [pair[1] for pair in feature_importances[:20]]
            x_values = list(range(len(importances20)))  # Make a bar chart
            plt.bar(x_values, importances20, orientation='vertical')  # Tick labels for x axis
            plt.xticks(x_values, features20, rotation='vertical')  # Axis labels and title
            plt.ylabel('Importance')
            plt.xlabel('Feature')
            plt.title('Top 20 Feature Importances')

            dir_name = os.path.abspath(save_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            fig_path = os.path.join(dir_name, '{}_{}_feature_importances.pdf'.format(self.config['model'], self.config['dataset']))
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            return True
        else:
            self.logger.info('To plot, the feature importance function is required to be implemented.')
            return False





