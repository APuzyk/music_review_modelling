from torch import from_numpy
import re
import numpy as np


class Predictor:
    def __init__(self, trainer):
        self.word_dict = trainer.review_catalogue.review_data_dict['word_dict']
        self.model = trainer.model
    
    def get_prediction(self, text: str):
        text_vec = self.get_text_vector(text)
        preds = self.model(from_numpy(text_vec))
        return [float(p) for p in preds[0]]
    
    def get_text_vector(self, text):
        text = self.clean_text(text)
        #TODO: how do we handle unknown values
        indexes = [self.word_dict.get(word, 0) for word in text]
        indexes = self.vectorize_text(indexes)
        indexes = np.array([indexes]) # add dim for batch

        return indexes
        
    
    def clean_text(self, text):
        text = re.sub(r'<.*?>', '', text) #html tags
        text = re.sub(r'[^\w\s]', '', text) # punctuation
        text = text.lower()
        text = text.strip()
        return text

    def vectorize_text(self, word_indexes):
        if len(word_indexes) > self.model.text_input_size:
            start = len(word_indexes) - self.model.text_input_size
            word_indexes = word_indexes[start:]
        else:
            [word_indexes.append(0) for _ in range(self.model.text_input_size - len(word_indexes))]
        
        return word_indexes
