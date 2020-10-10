import torch.nn as nn
from torch import from_numpy, flatten, cat, tanh
from .text_nn import TextNN


class TextLSTM(TextNN):

    def __init__(self, text_input_size, embedding_mat, hidden_dim=64, use_cuda=False):
        super(TextLSTM, self).__init__()

        self.use_cuda = use_cuda
        self.get_device()

        self.text_input_size = text_input_size

        # create layers
        self.embedding = nn.Embedding(embedding_mat.shape[0], embedding_mat.shape[1])
        self.create_embedding_layer(from_numpy(embedding_mat))
        self.embedding.weight.data.copy_(from_numpy(embedding_mat).to(device=self.device))

        self.lstm = nn.LSTM(input_size=int(self.embedding.embedding_dim), 
                            hidden_size=hidden_dim, 
                            num_layers=2, 
                            bidirectional=True)

        

        self.fc1 = nn.Linear(hidden_dim, 2)
        self.softmax = nn.Softmax(dim=1)

        self.model_type = 'TextLSTM'

    def create_embedding_layer(self, embedding_mat):
        self.embedding = nn.Embedding(embedding_mat.shape[0], embedding_mat.shape[1])
        self.embedding.weight.data.copy_(embedding_mat)
        self.embedding.requires_grad_(requires_grad=False)

    def forward(self, x):
        batch_size = x.shape[0]
        x.to(device=self.device)
        x = self.embedding(x)
        x = x.view(batch_size, 1, self.text_input_size, self.embedding.embedding_dim)
        
        x = self.lstm(x)
        x = tanh(self.fc1(x))
        x = self.softmax(x)

        return x
