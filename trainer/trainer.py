from reviews.review_catalog import ReviewCatalogue
from models.model_factory import ModelFactory
import os


class Trainer:

    def __init__(self):
        self.review_catalogue = ReviewCatalogue()
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
                                  'ngram_filters': [2, 3, 4]})

        self.model = mf.build_model()

        self.model.train_model(self.review_catalogue.get_train_content(),
                               self.review_catalogue.get_train_metadata(),
                               self.review_catalogue.get_train_y())

        self.get_predictions()
        self.save_predictions()

    def get_predictions(self):
        self.train_y = self.review_catalogue.get_train_y()[:, 1].tolist()
        self.holdout_y = self.review_catalogue.get_holdout_y()[:, 1].tolist()

        self.train_y_hat = self.model.predict(self.review_catalogue.get_train_content(),
                                              self.review_catalogue.get_train_metadata()).tolist()

        self.holdout_y_hat = self.model.predict(self.review_catalogue.get_holdout_content(),
                                                self.review_catalogue.get_holdout_metadata()).tolist()

    def save_predictions(self, dir='/home/apuzyk/Projects/music_reviews/data'):
        train_file = os.path.join(dir, str(self.review_catalogue.uuid) + '_' + 'train_predictions.csv')
        holdout_file = os.path.join(dir, str(self.review_catalogue.uuid) + '_' + 'holdout_predictions.csv')

        with open(train_file, 'w+') as f:
            f.write('y,y_hat\n')
            for i in range(len(self.train_y_hat)):
                f.write(str(self.train_y[i]) + ',' + str(self.train_y_hat[i]) + '\n')

        with open(holdout_file, 'w+') as f:
            f.write('y,y_hat\n')
            for i in range(len(self.holdout_y_hat)):
                f.write(str(self.holdout_y[i]) + ',' + str(self.holdout_y_hat[i]) + '\n')
