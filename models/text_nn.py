class TextNN:

    def __init__(self):
        self.model = None

    def build_model(self, params):
        pass

    def train_model(self, train_features, train_y, epochs=10, validation_split=0.2):
        self.model.fit(train_features,
                       train_y,
                       epochs=epochs,
                       validation_split=validation_split)

    def predict(self, predict_data):
        return self.model.predict(predict_data)
