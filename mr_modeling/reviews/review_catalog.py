import logging
import os
import pickle
import random

import numpy as np

from ..interfaces.sqllite_interface import MusicReviewInterface
from ..interfaces.word_vec_interface import WordVecInterface

module_logger = logging.getLogger(__name__)


class ReviewCatalogue:

    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.is_test = config.is_test
        self.save = config.save_review_data_dict
        self.review_data_dict_file = config.review_data_dict_file
        self.save_dir = config.save_dir

        self.interface = MusicReviewInterface(c_type=config.music_review_type,
                                              loc=config.music_review_fn)
        self.w2v_interface = WordVecInterface(loc=config.w2v_fn)

        self.review_data_dict = dict()
        self.review_data_dict['uuid'] = self.interface.get_latest_uuid()
        self.review_data_dict['review_content'] = None
        self.review_data_dict['word_dict'] = None
        self.review_data_dict['review_metadata'] = None
        self.review_data_dict['word_mat'] = None
        self.review_data_dict['review_ids'] = None
        self.review_data_dict['content_mat'] = []
        self.review_data_dict['y'] = []
        self.review_data_dict['one_hot_lus'] = {}
        self.review_data_dict['metadata_mat'] = []
        self.review_data_dict['training_indices'] = {}

    def prepare_reviews(self):
        if self.review_data_dict_file:
            self.load_from_file()
        else:
            self.preprocess_reviews()

    def load_from_file(self):
        self.logger.info(f"Loading reviews from file {self.review_data_dict_file}...")
        self.review_data_dict = pickle.load(open(self.review_data_dict_file, "rb"))

    def preprocess_reviews(self):
        self.logger.info("pulling review data")
        self.pull_review_data()
        self.logger.info("creating_word_mat")
        self.create_word_mat()
        self.logger.info("Getting_review_ids")
        self.get_review_ids()
        self.logger.info("getting_outcome")
        self.get_outcome()
        self.logger.info("creating content_mat")
        self.create_content_mat()
        self.logger.info("creating_content_mat")
        self.create_metadata_mat()
        self.logger.info("Splitting_data")
        self.split_data()
        if self.save:
            self.write_review_data_dict_to_file()

    def pull_review_data(self):
        self.pull_review_content()
        self.pull_word_dict()
        self.pull_review_metadata()

    def pull_review_content(self):
        self.review_data_dict['review_content'] = self.interface.pull_music_review_text(self.review_data_dict['uuid'],
                                                                                        self.is_test)

    def pull_word_dict(self):
        self.review_data_dict['word_dict'] = self.interface.pull_word_dict(self.review_data_dict['uuid'])

    def pull_review_metadata(self):
        self.review_data_dict['review_metadata'] = self.interface.pull_review_metadata(self.review_data_dict['uuid'],
                                                                                       self.is_test)

    def create_word_mat(self):
        self.review_data_dict['word_mat'] = self.w2v_interface.load_vecs_for_dict(self.review_data_dict['word_dict'])

    def get_review_ids(self):
        self.review_data_dict['review_ids'] = [i for i in self.review_data_dict['review_metadata']]

    def get_outcome(self):
        for i in self.review_data_dict['review_ids']:
            self.review_data_dict['y'].append(self.review_data_dict['review_metadata'][i]['score'])

        self.review_data_dict['y'] = [int(i > np.mean(self.review_data_dict['y'])) for i in self.review_data_dict['y']]
        self.review_data_dict['y'] = np.array(self.review_data_dict['y'])

    def create_content_mat(self, max_len_quantile=0.99):

        max_len = int(np.quantile([len(i) for i in self.review_data_dict['review_content'].values()], max_len_quantile))

        for i in self.review_data_dict['review_ids']:
            vec = self.review_data_dict['review_content'][i]
            if len(vec) >= max_len:
                start_position = len(vec) - max_len
                vec = vec[start_position:]
            else:
                [vec.append(0) for i in range(max_len - len(vec))]

            self.review_data_dict['content_mat'].append(vec)

        self.review_data_dict['content_mat'] = np.array(self.review_data_dict['content_mat'])

    def create_metadata_mat(self):
        # deal with strings
        strings = ['author_type', 'genre']
        data = []

        for string in strings:
            l = []
            for j in self.review_data_dict['review_ids']:
                l.append(self.review_data_dict['review_metadata'][j][string])

            self.review_data_dict['one_hot_lus'][string], mat = self.get_onehot(l)
            data.append(mat)

        # deal with nums
        metadata_mat = []
        for i in self.review_data_dict['review_ids']:
            to_add = self.review_data_dict['review_metadata'][i]
            metadata_mat.append([to_add['pub_month'],
                                 to_add['pub_weekday'],
                                 to_add['year'],
                                 to_add['pub_year'],
                                 to_add['pub_day']])

        metadata_mat = np.array(metadata_mat)
        data.append(metadata_mat)
        self.review_data_dict['metadata_mat'] = np.concatenate(data, axis=1)

    @staticmethod
    def get_onehot(l):
        lu = dict((c, i) for i, c in enumerate(list(set(l))))
        string_indices = []
        for i in l:
            string_indices.append(lu[i])
        string_indices = np.array(string_indices)
        one_hot = np.zeros((string_indices.size, string_indices.max() + 1))
        one_hot[np.arange(string_indices.size), string_indices] = 1
        return lu, one_hot

    def split_data(self):
        assert (self.review_data_dict['metadata_mat'].shape[0] == self.review_data_dict['content_mat'].shape[0])
        assert (self.review_data_dict['metadata_mat'].shape[0] == self.review_data_dict['y'].shape[0])

        n = self.review_data_dict['metadata_mat'].shape[0]

        self.review_data_dict['training_indices']['train'] = random.sample(range(n), int(n * 0.8))
        self.review_data_dict['training_indices']['holdout'] = [i for i in range(n) if
                                                                i not in self.review_data_dict['training_indices'][
                                                                    'train']]

    def write_review_data_dict_to_file(self):
        rc_loc = os.path.join(self.save_dir, "review_data_dict.p")
        self.logger.info(f"Saving review catalogue to {rc_loc}")
        pickle.dump(self.review_data_dict,
                    open(rc_loc, "wb"))

    def get_train_metadata(self):
        return self.review_data_dict['metadata_mat'][self.review_data_dict['training_indices']['train'], :]

    def get_holdout_metadata(self):
        return self.review_data_dict['metadata_mat'][self.review_data_dict['training_indices']['holdout'], :]

    def get_train_content(self):
        return self.review_data_dict['content_mat'][self.review_data_dict['training_indices']['train'], :]

    def get_holdout_content(self):
        return self.review_data_dict['content_mat'][self.review_data_dict['training_indices']['holdout'], :]

    def get_train_y(self):
        return self.review_data_dict['y'][self.review_data_dict['training_indices']['train']]

    def get_holdout_y(self):
        return self.review_data_dict['y'][self.review_data_dict['training_indices']['holdout']]

    def get_text_length(self):
        return self.review_data_dict['content_mat'].shape[1]

    def get_word_mat(self):
        return self.review_data_dict['word_mat']

    def get_metadata_dim(self):
        return self.review_data_dict['metadata_mat'].shape[1]
