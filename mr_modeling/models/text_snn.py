# from keras.layers import Input, Dense, Dropout
# from keras import Model
# from models.text_nn import TextNN
#
#
# # We're going super simple here to build something that just uses our meta data
# # I want to see if this is performant on its own and what the issue might be from the
# # deep and wide model
# class TextSNN(TextNN):
#
#     def __init__(self, wide_feature_num):
#         super(TextSNN, self).__init__()
#         self.wide_feature_num = wide_feature_num
#
#         self.model_type = 'TextSNN'
#         self.build_model()
#
#     def build_model(self):
#         wide_input = Input(shape=(self.wide_feature_num,))
#         wide_data = Dense(15)(wide_input)
#         wide_data = Dropout(rate=0.2)(wide_data)
#
#         predictions = Dense(2, activation='softmax')(wide_data)
#
#         model = Model(inputs=wide_input, outputs=predictions)
#
#         # adam = Adam(lr=0.01)
#         model.compile(optimizer='adadelta', loss='categorical_crossentropy')
#         self.model = model
