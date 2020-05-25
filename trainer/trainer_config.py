import json
import os
from time import time


class DataConfig:
    def __init__(self, config, base_dir, is_test):
        self.is_test = is_test
        self.music_review_fn = os.path.join(base_dir, config['music_review_fn'])
        self.music_review_type = config['music_review_type']
        self.w2v_fn = os.path.join(base_dir, config['w2v_fn'])


class ModelConfig:
    def __init__(self, config, is_test):
        self.val_split = config['val_split']
        self.holdout_split = config['holdout_split']
        self.model_type = config['model_type']
        self.ngram_filters = config['ngram_filters']
        self.is_test = is_test


class TrainerConfig:
    def __init__(self, config, is_test, time_id="".format(int(time()))):
        self.is_test = is_test
        with open(config, 'r') as f:
            config = json.loads(f)

        self.base_dir = config['base_dir']
        self.time_id = time_id
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.model_dir = os.path.join(self.base_dir, 'models')
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.data_config = DataConfig(config['data_config'], self.data_dir, is_test)
        self.model_config = ModelConfig(config['model_config'], is_test)

        for i in [self.base_dir, self.data_dir, self.model_dir, self.log_dir]:
            if not os.path.isdir(i):
                os.mkdir(i)
