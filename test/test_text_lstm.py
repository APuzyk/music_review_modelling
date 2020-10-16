import unittest
import numpy as np
from mr_modeling.models.text_lstm import TextLSTM


class TestTextCNN(unittest.TestCase):
    def test_init(self):
        self.assertIsInstance(TextLSTM(text_input_size=100,
                                       embedding_mat=np.random.rand(10, 300),
                                       hidden_dim=32,
                                       use_cuda=False,
                                       bidirectional=True),
                              TextLSTM)
