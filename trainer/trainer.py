from reviews.review_catalog import ReviewCatalogue
from models.model_factory import ModelFactory
from trainer.trainer_config import TrainerConfig
import os


class Trainer:

    def __init__(self, config):
        self.config = TrainerConfig(config)
        self.review_catalogue = ReviewCatalogue(self.config.data_config)
        self.model = None
        self.train_y_hat = None
        self.train_y = None
        self.holdout_y_hat = None
        self.holdout_y = None

    def train_model(self):
        self.review_catalogue.preprocess_reviews()
        mf = ModelFactory(params={'content_mat':self.review_catalogue.content_mat,
                                  'embedding_mat': self.review_catalogue.word_mat,
                                  'wide_features':self.review_catalogue.metadata_mat,
                                  'ngram_filters': self.config.model_config.ngram_filters},
                          model_type=self.config.model_config.model_type)

        self.model = mf.build_model()

        self.model.train_model(self.get_feature_data()['train_features'],
                               self.review_catalogue.get_train_y())

        self.get_predictions()
        self.save_predictions(self.config.data_dir)

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
        elif self.model.model_type == "TextCNN":
            return {'train_features': self.review_catalogue.get_train_content(),
                    'holdout_features': self.review_catalogue.get_holdout_content()}
        else:
            raise NotImplementedError("Model type {} not implemented".format(self.model.model_type))

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
