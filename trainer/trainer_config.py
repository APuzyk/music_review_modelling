import yaml
import os


class DataConfig:
    def __init__(self, config, base_dir):
        self.music_review_fn = os.path.join(base_dir, config['music_review_fn'])
        self.music_review_type = config['music_review_type']
        self.w2v_fn = os.path.join(base_dir, config['w2v_fn'])


class ModelConfig:
    def __init__(self, config):
        self.val_split = config['val_split']
        self.holdout_split = config['holdout_split']
        self.model_type = config['model_type']
        self.ngram_filters = config['ngram_filters']


class TrainerConfig:
    def __init__(self, config):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)

        self.base_dir = config['base_dir']
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.data_config = DataConfig(config['data_config'], self.data_dir)
        self.model_config = ModelConfig(config['model_config'])

