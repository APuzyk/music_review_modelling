from reviews.review_catalog import ReviewCatalogue
from models.model_factory import ModelFactory
from trainer.trainer_config import TrainerConfig
from sklearn.metrics import auc, precision_recall_curve
from helpers.helpers import sort_l_x_by_l_y
import os
import json


class Trainer:

    def __init__(self, config, is_test=False):
        self.config = TrainerConfig(config, is_test)
        self.review_catalogue = ReviewCatalogue(self.config.data_config)
        self.model = None
        self.train_y_hat = None
        self.train_y = None
        self.holdout_y_hat = None
        self.holdout_y = None
        self.performance_data = {}

    def train_model(self):
        self.review_catalogue.preprocess_reviews()
        mf = ModelFactory(params={'content_mat':self.review_catalogue.content_mat,
                                  'embedding_mat': self.review_catalogue.word_mat,
                                  'wide_features':self.review_catalogue.metadata_mat,
                                  'ngram_filters': self.config.model_config.ngram_filters},
                          model_config=self.config.model_config)

        self.model = mf.build_model()

        self.model.train_model(self.get_feature_data()['train_features'],
                               self.review_catalogue.get_train_y(),
                               self.config.model_config.is_test,
                               self.config)

        self.get_predictions()
        self.save_predictions(self.config.data_dir)
        self.get_performance_data()

    def get_predictions(self):
        self.train_y = self.review_catalogue.get_train_y()[:, 1].tolist()
        self.holdout_y = self.review_catalogue.get_holdout_y()[:, 1].tolist()

        input_data = self.get_feature_data()

        self.train_y_hat = self.model.predict(input_data['train_features']).tolist()
        self.holdout_y_hat = self.model.predict(input_data['holdout_features']).tolist()

    def get_feature_data(self):
        if self.model.model_type == "TextCNNWideAndDeep":
            return {'train_features': [self.review_catalogue.get_train_content(),
                                       self.review_catalogue.get_train_metadata()],
                    'holdout_features': [self.review_catalogue.get_holdout_content(),
                                         self.review_catalogue.get_holdout_metadata()]}
        elif self.model.model_type in ("TextCNN", "TextLSTM"):
            return {'train_features': self.review_catalogue.get_train_content(),
                    'holdout_features': self.review_catalogue.get_holdout_content()}
        elif self.model.model_type == "TextSNN":
            return {'train_features': self.review_catalogue.get_train_metadata(),
                    'holdout_features': self.review_catalogue.get_holdout_metadata()}
        else:
            raise NotImplementedError("Model type {} not implemented".format(self.model.model_type))

    def get_performance_data(self):
        train_y_hat = self.get_pos_values(self.train_y_hat)
        train_y = self.train_y
        holdout_y_hat = self.get_pos_values(self.holdout_y_hat)
        holdout_y = self.holdout_y
        self.performance_data['auc_train'] = auc(sorted(train_y_hat),
                                                 sort_l_x_by_l_y(train_y,
                                                                 train_y_hat))

        self.performance_data['auc_holdout'] = auc(sorted(holdout_y_hat),
                                                   sort_l_x_by_l_y(holdout_y,
                                                                   holdout_y_hat))

        self.performance_data['p_r_curve_train'] = self.create_p_r_dict(precision_recall_curve(train_y,
                                                                                               train_y_hat))

        self.performance_data['p_r_curve_holdout'] = self.create_p_r_dict(precision_recall_curve(holdout_y,
                                                                                                 holdout_y_hat))

        self.save_performance_data()

    @staticmethod
    def create_p_r_dict(prc):
        o = dict()
        o['precision'] = prc[0].tolist()
        o['recall'] = prc[1].tolist()
        o['threshold'] = prc[2].tolist()

        return o

    def save_performance_data(self):
        for k, v in self.performance_data.items():
            print("data for: " + k + "\n")
            print("\t" + str(v))

        o = json.dumps(self.performance_data)
        f = open(self.config.data_dir + "/performance_{}.json".format(self.config.time_id), "w+")
        f.write(o)
        f.close()

    @staticmethod
    def get_pos_values(l):
        return [i[1] for i in l]

    def save_predictions(self, dir):
        train_file = os.path.join(dir, str(self.review_catalogue.uuid) + '_' + 'train_predictions.csv')
        holdout_file = os.path.join(dir, str(self.review_catalogue.uuid) + '_' + 'holdout_predictions.csv')

        with open(train_file, 'w+') as f:
            f.write('y,y_hat\n')
            for i in range(len(self.train_y_hat)):
                f.write(str(self.train_y[i]) + ',' + str(self.train_y_hat[i][1]) + '\n')

        with open(holdout_file, 'w+') as f:
            f.write('y,y_hat\n')
            for i in range(len(self.holdout_y_hat)):
                f.write(str(self.holdout_y[i]) + ',' + str(self.holdout_y_hat[i][1]) + '\n')
