# coding: utf-8
# @Time   : 2021/05/24
# @Author : enoche
# @Email  : enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import ETADataset
from utils.dataloader import DataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model
from models.common.trainer import BatchTrainer, FullTrainer
import platform
import os


def quick_start(model, dataset, config_dict):
    # merge config dict
    config = Config(model, dataset, config_dict)
    save_model = config['save_model']
    init_logger(config)
    logger = getLogger()
    # print config info
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    # print config info.
    logger.info(config)
    # load data
    dataset = ETADataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split(config['valid_test_days'])
    logger.info('\n====Training===='+str(train_dataset))
    logger.info('\n====Validation===='+str(valid_dataset))
    logger.info('\n====Testing===='+str(test_dataset)+'\n======================\n\n')

    # wrap into dataloader
    train_data = DataLoader(config, train_dataset, additional_dataset=dataset,
                            batch_size=config['train_batch_size'], shuffle=config['shuffle'], training_mode=True)
    (valid_data, test_data) = (DataLoader(config, valid_dataset, additional_dataset=train_dataset,
                                          batch_size=config['eval_batch_size'], training_mode=False),
                               DataLoader(config, test_dataset, additional_dataset=train_dataset,
                                          batch_size=config['eval_batch_size'], training_mode=False))

    ############ loop
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 1e+07
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    ############ Dataset loaded, run model
    for hyper_tuple in combinators:
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        # random seed
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        train_data.reset_loader()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        # trainer loading and initialization
        trainer = BatchTrainer(config, model) if config['trainer_type'] == 'batch' else FullTrainer(config, model)
        # model training
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        # model evaluation
        test_result = trainer.evaluate(test_data, load_best_model=save_model, is_test=True, idx=idx+1)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, test_result))

        # save best test
        if test_result[val_metric] < best_test_value:
            best_test_value = test_result[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(best_valid_result))
        logger.info('test result: {}'.format(test_result))

        logger.info('=================================')
        logger.info('Current ██ BEST ██: \n\tParameters: {}={},'
                    '\n\tbest valid: {},\n\tbest test: {}\n\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   hyper_ret[best_test_idx][1],
                                                                   hyper_ret[best_test_idx][2]))
        #########
        if config['plot_enabled']:
            plotted = trainer.plot_feature_importance(train_data.feature_columns, config['plot_save_path'])
            if plotted:
                logger.info('\n\n=======Plots saved to: ' + config['plot_save_path'])
    #########

    logger.info('\n\n=======Summary===============')
    for (p, k, v) in hyper_ret:
        logger.info('\n\tParameters: {}={},\n\tbest valid: {},\n\tbest test: {}'.format(config['hyper_parameters'], p, k, v))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\n\tParameters: {}={},\n\tValid: {},\n\tTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   hyper_ret[best_test_idx][1],
                                                                   hyper_ret[best_test_idx][2]))


