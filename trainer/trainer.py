from reviews.review_catalog import ReviewCatalogue
from models.model_factory import ModelFactory


class Trainer:

    def __init__(self):
        self.review_catalogue = ReviewCatalogue()
        self.model = None

    def train_model(self):
        self.review_catalogue.preprocess_reviews()
        mf = ModelFactory(params={'content_mat':self.review_catalogue.content_mat,
                                  'embedding_mat': self.review_catalogue.word_mat,
                                  'wide_features':self.review_catalogue.metadata_mat,
                                  'ngram_filters': [2, 3, 4]})

        self.model = mf.build_model()
