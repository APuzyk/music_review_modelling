import torch.nn as nn
from torch import from_numpy, flatten, cat, tanh
from .text_nn import TextNN


class TextLSTM(TextNN):

    def __init__(self, text_input_size, embedding_mat, hidden_dim=64, use_cuda=False,
                bidirectional=False):
        super(TextLSTM, self).__init__()

        self.use_cuda = use_cuda
        self.get_device()

        self.text_input_size = text_input_size
        self.hidden_dim = hidden_dim

        # create layers
        self.embedding = nn.Embedding(embedding_mat.shape[0], embedding_mat.shape[1])
        self.create_embedding_layer(from_numpy(embedding_mat))
        self.embedding.weight.data.copy_(from_numpy(embedding_mat).to(device=self.device))

        self.lstm = nn.LSTM(input_size=int(self.embedding.embedding_dim), 
                            hidden_size=self.hidden_dim, 
                            num_layers=2, 
                            bidirectional=bidirectional,
                            batch_first=True)

        
        multiple = 2 if bidirectional else 1
        self.fc1 = nn.Linear(hidden_dim * multiple * self.text_input_size, self.text_input_size)
        self.fc2 = nn.Linear(self.text_input_size, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 2)
        self.softmax = nn.Softmax(dim=1)

        self.model_type = 'TextLSTM'

    def create_embedding_layer(self, embedding_mat):
        #TODO: Figure out how to handle padding dimension
        self.embedding = nn.Embedding(embedding_mat.shape[0], embedding_mat.shape[1])
        self.embedding.weight.data.copy_(embedding_mat)
        self.embedding.requires_grad_(requires_grad=False)

    def forward(self, x):
        batch_size = x.shape[0]
        x.to(device=self.device)
        x = self.embedding(x)
        x = x.view(batch_size, self.text_input_size, self.embedding.embedding_dim)

        x, _ = self.lstm(x)
       
        x = x.reshape(batch_size, -1)

        x = tanh(self.fc1(x))
        x = tanh(self.fc2(x))
        x = tanh(self.fc3(x))
        x = self.softmax(x)

        return x
