from interfaces.sqllite_interface import MusicReviewInterface
from interfaces.word_vec_interface import WordVecInterface
import numpy as np
import random
import logging

module_logger = logging.getLogger(__name__)


class ReviewCatalogue:

    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.is_test = config.is_test
        self.interface = MusicReviewInterface(c_type=config.music_review_type, loc=config.music_review_fn)
        self.w2v_interface = WordVecInterface(loc=config.w2v_fn)
        self.uuid = self.interface.get_latest_uuid()
        self.review_content = None
        self.word_dict = None
        self.review_metadata = None
        self.word_mat = None
        self.review_ids = None
        self.content_mat = []
        self.y = []
        self.one_hot_lus = {}
        self.metadata_mat = []
        self.training_indices = {}

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

    def pull_review_data(self):
        self.pull_review_content()
        self.pull_word_dict()
        self.pull_review_metadata()

    def pull_review_content(self):
        self.review_content = self.interface.pull_music_review_text(self.uuid, self.is_test)

    def pull_word_dict(self):
        self.word_dict = self.interface.pull_word_dict(self.uuid)

    def pull_review_metadata(self):
        self.review_metadata = self.interface.pull_review_metadata(self.uuid, self.is_test)

    def create_word_mat(self):
        self.word_mat = self.w2v_interface.load_vecs_for_dict(self.word_dict)

    def get_review_ids(self):
        self.review_ids = [i for i in self.review_metadata]

    def get_outcome(self):
        for i in self.review_ids:
            self.y.append(self.review_metadata[i]['score'])

        self.y = [int(i > np.mean(self.y)) for i in self.y]
        self.y = np.array(self.y)

    def create_content_mat(self, max_len_quantile=0.99):

        max_len = int(np.quantile([len(i) for i in self.review_content.values()], max_len_quantile))

        for i in self.review_ids:
            vec = self.review_content[i]
            if len(vec) >= max_len:
                start_position = len(vec) - max_len
                vec = vec[start_position:]
            else:
                [vec.append(0) for i in range(max_len - len(vec))]

            self.content_mat.append(vec)

        self.content_mat = np.array(self.content_mat)

    def create_metadata_mat(self):
        # deal with strings
        strings = ['author_type', 'genre']
        data = []

        for string in strings:
            l = []
            for j in self.review_ids:
                l.append(self.review_metadata[j][string])

            self.one_hot_lus[string], mat = self.get_onehot(l)
            data.append(mat)

        # deal with nums
        metadata_mat = []
        for i in self.review_ids:
            to_add = self.review_metadata[i]
            metadata_mat.append([to_add['pub_month'],
                                 to_add['pub_weekday'],
                                 to_add['year'],
                                 to_add['pub_year'],
                                 to_add['pub_day']])

        metadata_mat = np.array(metadata_mat)
        data.append(metadata_mat)
        self.metadata_mat = np.concatenate(data, axis=1)

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
        assert(self.metadata_mat.shape[0] == self.content_mat.shape[0])
        assert(self.metadata_mat.shape[0] == self.y.shape[0])

        n = self.metadata_mat.shape[0]

        self.training_indices['train'] = random.sample(range(n), int(n*0.8))
        self.training_indices['holdout'] = [i for i in range(n) if i not in self.training_indices['train']]

    def get_train_metadata(self):
        return self.metadata_mat[self.training_indices['train'], :]

    def get_holdout_metadata(self):
        return self.metadata_mat[self.training_indices['holdout'], :]

    def get_train_content(self):
        return self.content_mat[self.training_indices['train'], :]

    def get_holdout_content(self):
        return self.content_mat[self.training_indices['holdout'], :]

    def get_train_y(self):
        return self.y[self.training_indices['train']]

    def get_holdout_y(self):
        return self.y[self.training_indices['holdout']]

