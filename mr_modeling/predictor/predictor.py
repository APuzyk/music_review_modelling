from trainer.trainer import Trainer
import re
import numpy as np


class Predictor:
    def __init__(self, trainer: Trainer):
        self.word_dict = trainer.review_catalogue.review_data_dict['word_dict']
        self.model = trainer.model
    
    def get_prediction(self, textg: str):
        text_vec = self.get_text_vector(text)
    
    def get_text_vector(self, text):
        text = self.clean_text(text)
        #TODO: how do we handle unknown values
        indexs = [self.word_dict.get(word, 0) for word in text]

        return self.vectorize_text(indexs)
        
    
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
        
        return np.array(word_indexes)



if len(vec) >= max_len:
                start_position = len(vec) - max_len
                vec = vec[start_position:]
            else:
                [vec.append(0) for i in range(max_len - len(vec))]
