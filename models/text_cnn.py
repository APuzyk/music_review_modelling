import torch.nn as nn
from torch import from_numpy, flatten, cat, tanh
from models.text_nn import TextNN


class TextCNN(TextNN):

    def __init__(self, text_input_size, embedding_mat, ngram_filters=[3, 4, 5]):
        super(TextCNN, self).__init__()

        self.text_input_size = text_input_size

        # create layers
        self.embedding = nn.Embedding(embedding_mat.shape[0], embedding_mat.shape[1])
        self.create_embedding_layer(from_numpy(embedding_mat))

        self.ngram_filters = ngram_filters
        c2d_out_dim = self.create_conv_layers(ngram_filters)

        self.fc1 = nn.Linear(c2d_out_dim, 100)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(100, 2)

        self.model_type = 'TextCNN'

    def create_embedding_layer(self, embedding_mat):
        self.embedding = nn.Embedding(embedding_mat.shape[0], embedding_mat.shape[1])
        self.embedding.weight.data.copy_(embedding_mat)
        self.embedding.requires_grad_(requires_grad=False)

    def create_conv_layers(self, ngram_filters):
        out_dim = 0
        for i in ngram_filters:
            self.__setattr__(f"c2d_ngram_{i}",
                             nn.Conv2d(1, 300,
                                       kernel_size=(i, int(self.embedding.embedding_dim)),
                                       stride=(1, int(self.embedding.embedding_dim))))
            self.__setattr__(f"pool_ngram_{i}",
                             nn.MaxPool2d(stride=(300, 1), kernel_size=(300, 1)))
            out_dim += self.text_input_size - i + 1
        return out_dim

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.embedding(x)
        x = x.view(batch_size, 1, self.text_input_size, self.embedding.embedding_dim)
        layers = []
        for i in self.ngram_filters:
            l = self.__getattr__(f"c2d_ngram_{i}")(x)
            l = tanh(l)
            l = l.permute(0, 3, 1, 2)
            l = self.__getattr__(f"pool_ngram_{i}")(l)
            l = flatten(l, 1)
            layers.append(l)
        x = cat(layers, 1)
        x = tanh(self.fc1(x))
        x = self.dropout(x)
        x = tanh(self.fc2(x))

        return x
