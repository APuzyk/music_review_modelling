from keras.layers import Input, Embedding, Conv2D, MaxPool2D, Reshape

class text_cnn_wide_and_deep:

    def __init__(self, text_input_size, ngram_filters=[3, 4. 5], dict_size):
        self.text_input_size = text_input_size
        self.ngram_filters = ngram_filters
        self.model = None
        self.dict_size = dict_size

    def model(self):

        input = Input(self.text_input_size)

        word_mat = Embedding(self.dict_size, 300, input_length=self.text_input_size)(input)

        x_3d = Reshape(target_shape=(self.text_input_size, 300, ))(word_mat)
        conv_layers = []

        for i in range(len(self.ngram_filters)):
            x = Conv2D(filters=300, kernel_size=(self.ngram_filters[i], 300), strides=(1, ))(x_3d)

            x = MaxPool2D()
            conv_layers.append()
        x = Conv2D(self.ngram_filters, )


