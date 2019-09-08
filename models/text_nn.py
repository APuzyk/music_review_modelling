from keras.callbacks import TensorBoard, ModelCheckpoint
from time import time


class TextNN:
    def __init__(self):
        self.model = None

    def build_model(self, params):
        pass

    def train_model(self, train_features, train_y, is_test, config, epochs=100, validation_split=0.2):
        if is_test:
            epochs = 5

        model_time = "/{}".format(int(time()))
        tensorboard = TensorBoard(log_dir=config.log_dir + model_time)
        checkpoint = ModelCheckpoint(config.model_dir + model_time + '.hdf5', verbose=1, save_best_only=True)

        self.model.fit(train_features,
                       train_y,
                       epochs=epochs,
                       validation_split=validation_split,
                       callbacks=[tensorboard, checkpoint])

    def predict(self, predict_data):
        return self.model.predict(predict_data)
