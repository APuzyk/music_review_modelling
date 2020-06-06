from .text_cnn_wide_and_deep import TextCNNWideAndDeep
from .text_cnn import TextCNN
#from models.text_snn import TextSNN
#from models.text_lstm import TextLSTM


class ModelFactory:
    def __init__(self, params, model_config):
        self.model_type = model_config.model_type
        self.params = params
        self.use_cuda = model_config.use_cuda
        self.ngram_filters = model_config.ngram_filters

    def build_model(self):
        if self.model_type == 'TextCNNWideAndDeep':
            model = TextCNNWideAndDeep(text_input_size=self.params['text_length'],
                                       embedding_mat=self.params['embedding_mat'],
                                       wide_feature_num=self.params['wide_feature_num'],
                                       ngram_filters=self.ngram_filters,
                                       use_cuda=self.use_cuda)

        elif self.model_type == 'TextCNN':
            model = TextCNN(text_input_size=self.params['text_length'],
                            embedding_mat=self.params['embedding_mat'],
                            ngram_filters=self.ngram_filters,
                            use_cuda=self.use_cuda)
        # elif self.model_type == 'TextSNN':
        #     raise NotImplementedError
        #     model = TextSNN(wide_feature_num=self.params['wide_features'].shape[1])
        # elif self.model_type == 'TextLSTM':
        #     model = TextLSTM(text_input_size=self.params['content_mat'].shape[1],
        #                      embedding_mat=self.params['embedding_mat'])
        else:
            raise NotImplementedError("Models of type %s are not yet implemented" % self.model_type)

        return model

