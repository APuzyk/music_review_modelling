import unittest
import numpy as np
from mr_modeling.models.text_cnn import TextCNN


class TestTextCNN(unittest.TestCase):
    def test_init(self):
        self.assertIsInstance(TextCNN(text_input_size=100,
                                      embedding_mat=np.random.rand(10, 300),
                                      ngram_filters=[3, 4, 5],
                                      use_cuda=False),
                              TextCNN)
