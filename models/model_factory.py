from models.text_cnn_wide_and_deep import TextCNNWideAndDeep
from models.text_cnn import TextCNN


class ModelFactory:
    def __init__(self, params, model_config):
        self.model_type = model_config.model_type
        self.params = params

    def build_model(self):
        if self.model_type == 'TextCNNWideAndDeep':
            model = TextCNNWideAndDeep(text_input_size=self.params['content_mat'].shape[1],
                                       embedding_mat=self.params['embedding_mat'],
                                       wide_feature_num=self.params['wide_features'].shape[1],
                                       ngram_filters=self.params['ngram_filters'])

        elif self.model_type == 'TextCNN':
            model = TextCNN(text_input_size=self.params['content_mat'].shape[1],
                            embedding_mat=self.params['embedding_mat'],
                            ngram_filters=self.params['ngram_filters'])
        else:
            raise NotImplementedError("Models of type %s are not yet implemented" % self.model_type)

        return model

