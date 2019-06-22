from .text_cnn import TextCNNWideAndDeep


class ModelFactory:
    def __init__(self, model_type='TextCNNWideAndDeep', params):
        self.model_type = model_type
        self.params = params

    def build_model(self):
        if self.model_type == 'TextCnnWideAndDeep':
            model = TextCNNWideAndDeep(text_input_size=self.params['text_mat'].shape[1],
                                       embedding_mat=self.params['embedding_mat'],
                                       wide_feature_num=self.params['wide_features'].shape[1],
                                       ngram_filters=self.params['ngram_filters'])
        else:
            raise NotImplemented "Models of type %s are not yet implemented" % self.model_type

        return model

