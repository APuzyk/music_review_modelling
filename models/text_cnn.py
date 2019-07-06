from keras.layers import Input, Embedding, Conv2D, MaxPooling2D, Reshape, Flatten, Dense, Concatenate, Dropout
from keras import Model


class TextCNNWideAndDeep:

    def __init__(self, text_input_size, embedding_mat, wide_feature_num, ngram_filters=[3, 4, 5]):
        self.text_input_size = text_input_size
        self.ngram_filters = ngram_filters
        self.embedding_mat = embedding_mat
        self.wide_feature_num = wide_feature_num

        self.model=None
        self.build_model()

    def build_model(self):
        inputs = Input(shape=(self.text_input_size,))

        word_mat = Embedding(self.embedding_mat.shape[0],
                             self.embedding_mat.shape[1],
                             weights=[self.embedding_mat],
                             input_length=(self.text_input_size,), trainable=False)(inputs)

        x_3d = Reshape(target_shape=(self.text_input_size, self.embedding_mat.shape[1], 1))(word_mat)
        conv_layers = []

        for i in self.ngram_filters:
            x = Conv2D(filters=300, kernel_size=(i, int(word_mat.shape[2])), strides=(1, int(word_mat.shape[2])))(x_3d)
            x = Reshape((int(x.shape[1]), 300, 1))(x)
            x = MaxPooling2D(strides=(300, 1))(x)
            x = Flatten()(x)
            conv_layers.append(x)

        x = Concatenate()(conv_layers)

        x = Dense(100)(x)

        wide_data = Input(shape=(self.wide_feature_num,))

        all_data = Concatenate()([x, wide_data])

        all_data = Dropout(rate=0.2)(all_data)

        predictions = Dense(2, activation='softmax')(all_data)

        model = Model(inputs=[inputs, wide_data], outputs=predictions)

        model.compile(optimizer="adam", loss='categorical_crossentropy')
        self.model = model

    def train_model(self, text, wide_data, y, epochs=10, validation_split=0.2):
        self.model.fit([text, wide_data],
                       y,
                       epochs=epochs,
                       validation_split=validation_split)

    def predict(self, text, wide_data):
        return self.model.predict([text, wide_data])
