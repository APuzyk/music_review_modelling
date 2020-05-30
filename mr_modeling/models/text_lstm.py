# from keras.layers import Input, Embedding, SpatialDropout1D, LSTM, Dense
# from keras import Model
# from models.text_nn import TextNN
#
#
# class TextLSTM(TextNN):
#
#     def __init__(self, text_input_size, embedding_mat):
#         super(TextLSTM, self).__init__()
#
#         self.text_input_size = text_input_size
#         self.embedding_mat = embedding_mat
#
#         self.model_type = 'TextLSTM'
#         self.build_model()
#
#     def build_model(self):
#         inputs = Input(shape=(self.text_input_size,))
#
#         word_mat = Embedding(self.embedding_mat.shape[0],
#                              self.embedding_mat.shape[1],
#                              weights=[self.embedding_mat],
#                              input_length=(self.text_input_size,), trainable=False)(inputs)
#
#         word_mat = SpatialDropout1D(0.2)(word_mat)
#
#         word_mat = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(word_mat)
#
#         predictions = Dense(2, activation='softmax')(word_mat)
#
#         model = Model(inputs=inputs, outputs=predictions)
#
#         model.compile(optimizer='adam', loss='categorical_crossentropy')
#         self.model = model
