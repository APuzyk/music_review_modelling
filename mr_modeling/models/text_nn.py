import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch import from_numpy


class TextNN(nn.Module):
    def __init__(self):
        super(TextNN, self).__init__()
        self.model = None

    def build_model(self, params):
        pass

    def train_model(self, train_features, train_y, epochs=100, batch_size=64):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters())
        for epoch in range(epochs):
            running_loss = 0.0
            train_features = np.array_split(train_features, int(train_features.shape[0]/batch_size) + 1)
            train_y = np.array_split(train_y, int(train_y.shape[0]/batch_size) + 1)
            i = 0
            for x, y in zip(train_features, train_y):
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(from_numpy(x))
                loss = criterion(outputs, from_numpy(y))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
                i += 1

    def predict(self, predict_data):
        pass
