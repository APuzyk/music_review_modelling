import unittest
import numpy as np
from mr_modeling.models.text_cnn_wide_and_deep import TextCNNWideAndDeep


class TestTextCNNWideAndDeep(unittest.TestCase):
    def test_init(self):
        self.assertIsInstance(TextCNNWideAndDeep(text_input_size=100,
                                                 embedding_mat=np.random.rand(10, 300),
                                                 wide_feature_num=20,
                                                 ngram_filters=[3, 4, 5],
                                                 use_cuda=False),
                              TextCNNWideAndDeep)
