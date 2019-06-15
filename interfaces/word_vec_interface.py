from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import numpy as np


class WordVecInterface:
    def __init__(self, loc):
        self.loc = loc

    def load_vecs_for_dict(self, word_dict):
        wv = KeyedVectors.load_word2vec_format(datapath(self.loc), binary=True)
        word_mat = np.random.uniform(-0.25, 0.25, (len(word_dict) + 1, 300))
        for i in range(len(word_dict)):
            word_mat[i] = word_dict[i]

        return word_mat
