from gensim.models import KeyedVectors
import numpy as np


class WordVecInterface:
    def __init__(self, loc):
        self.loc = loc

    def load_vecs_for_dict(self, word_dict):
        wv = KeyedVectors.load_word2vec_format(self.loc, binary=True)
        word_mat = np.random.uniform(-0.25, 0.25, (len(word_dict) + 1, 300))
        for k, v in word_dict.items():
            try:
                vec = wv[k]
            except KeyError:
                continue

            word_mat[v] = vec

        return word_mat
