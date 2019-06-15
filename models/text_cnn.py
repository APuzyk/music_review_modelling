from keras.layers import Input, Embedding, Conv2D, MaxPool2D
from keras.backend import expand_dims

class text_cnn_wide_and_deep:

    def __init__(self, text_input_size, ngram_filters=[3, 4. 5], dict_size):
        self.text_input_size = text_input_size
        self.ngram_filters = ngram_filters
        self.model = None
        self.dict_size = dict_size

    def model(self):

        input = Input(self.text_input_size)

        word_mat = Embedding(self.dict_size, 300, input_length=self.text_input_size)(input)

        x_3d = expand_dims(word_mat, 3)
        conv_layers = []

        for i in range(len(self.ngram_filters)):
            x = Conv2D(filters=300, kernel_size=(self.ngram_filters[i], 300), strides=(1, 300))(x_3d)
            #<tf.Tensor 'conv2d_5/BiasAdd:0' shape=(?, 96, 1, 300) dtype=float32>
            #TODO add max pooling
            x = MaxPool2D()(x)
            conv_layers.append(x)

        #TODO concat layers together
        #TODO add dense layer to get size down to 100, 1
        # TODO concat meta data


        #TODO add softmax layer

