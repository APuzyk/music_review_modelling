from ..reviews.review_catalog import ReviewCatalogue
from ..models.model_factory import ModelFactory
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import from_numpy
from ..trainer.trainer_config import TrainerConfig
from sklearn.metrics import auc, precision_recall_curve, roc_curve, average_precision_score
from ..helpers.helpers import sort_l_x_by_l_y, one_hot
import os
import json
import logging
import matplotlib.pyplot as plt


class Trainer:

    def __init__(self, config):
        self.logger = logging.getLogger("mr_modeling.trainer.trainer.Trainer")
        print(self.logger)
        self.logger.info("Creating Trainer")
        self.config = config
        self.review_catalogue = ReviewCatalogue(self.config.data_config)
        self.model = None
        self.train_y_hat = []
        self.train_y = []
        self.holdout_y_hat = []
        self.holdout_y = []
        self.performance_data = {}

    def train_model(self):
        self.logger.info("Preprocessing reviews")
        self.review_catalogue.preprocess_reviews()
        self.logger.info("Creating Model Object")
        mf = ModelFactory(params={'content_mat': self.review_catalogue.content_mat,
                                  'embedding_mat': self.review_catalogue.word_mat,
                                  'wide_features': self.review_catalogue.metadata_mat,
                                  'ngram_filters': self.config.model_config.ngram_filters},
                          model_config=self.config.model_config)

        self.model = mf.build_model()
        self.logger.info("Training Model...")
        self.optimize_model()

        self.get_predictions()
        self.logger.info(f"Saving data in {self.config.data_dir_save}")
        self.save_predictions(self.config.data_dir_save)
        self.get_performance_data()

    def optimize_model(self):
        epochs = self.config.model_config.epochs
        batch_size = self.config.model_config.batch_size
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.model_config.lr)
        train_features = self.get_feature_data()['train_features']
        train_y = one_hot(self.review_catalogue.get_train_y())

        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch+1} of {epochs}.")
            self.logger.info("Shuffling for epoch...")
            permutation = np.random.permutation(train_y.shape[0])
            train_features = np.take(train_features, permutation, axis=0, out=train_features)
            train_y = np.take(train_y, permutation, axis=0, out=train_y)

            self.logger.info("Splitting for epoch")
            train_features_tensors = np.array_split(train_features, int(train_features.shape[0] / batch_size) + 1)
            train_features_tensors = [from_numpy(a) for a in train_features_tensors]
            train_y_tensors = np.array_split(train_y, int(train_y.shape[0] / batch_size) + 1)
            train_y_tensors = [from_numpy(a) for a in train_y_tensors]

            self.logger.info(f"Running {len(train_y_tensors)} batches...")
            running_loss = 0.0
            i = 0
            for x, y in zip(train_features_tensors, train_y_tensors):
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(x)
                loss = criterion(outputs, y.float())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                if i % 50 == 49:  # print every 50 mini-batches
                    self.logger.info('[%d, %5d] loss: %.3f' %
                                     (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0
                i += 1

    def get_predictions(self):
        self.model.eval()
        self.train_y = self.review_catalogue.get_train_y().tolist()
        self.holdout_y = self.review_catalogue.get_holdout_y().tolist()

        input_data = self.get_feature_data()
        batch_size = self.config.model_config.batch_size
        train_features = np.array_split(input_data['train_features'],
                                        int(input_data['train_features'].shape[0] / batch_size)
                                        + 1)
        holdout_features = np.array_split(input_data['holdout_features'],
                                          int(input_data['holdout_features'].shape[0] / batch_size)
                                          + 1)
        for batch in train_features:
            self.train_y_hat += self.model(from_numpy(batch)).tolist()
        for batch in holdout_features:
            self.holdout_y_hat += self.model(from_numpy(batch)).tolist()

    def get_feature_data(self):
        #TODO: Split out into a separate training data class
        if self.model.model_type == "TextCNNWideAndDeep":
            return {'train_features': [self.review_catalogue.get_train_content(),
                                       self.review_catalogue.get_train_metadata()],
                    'holdout_features': [self.review_catalogue.get_holdout_content(),
                                         self.review_catalogue.get_holdout_metadata()]}
        elif self.model.model_type in ("TextCNN", "TextLSTM"):
            return {'train_features': self.review_catalogue.get_train_content(),
                    'holdout_features': self.review_catalogue.get_holdout_content()}
        elif self.model.model_type == "TextSNN":
            return {'train_features': self.review_catalogue.get_train_metadata(),
                    'holdout_features': self.review_catalogue.get_holdout_metadata()}
        else:
            raise NotImplementedError("Model type {} not implemented".format(self.model.model_type))

    def get_performance_data(self):
        train_y_hat = self.get_pos_values(self.train_y_hat)
        train_y = self.train_y
        holdout_y_hat = self.get_pos_values(self.holdout_y_hat)
        holdout_y = self.holdout_y

        self.performance_data['p_r_curve_train'] = self.create_p_r_dict(precision_recall_curve(train_y,
                                                                                               train_y_hat))

        self.performance_data['p_r_curve_holdout'] = self.create_p_r_dict(precision_recall_curve(holdout_y,
                                                                                                 holdout_y_hat))
        self.calc_roc_and_auc()
        self.calc_precision_recall()
        self.save_performance_data()

    def calc_roc_and_auc(self):
        to_run = {'holdout': [one_hot(np.array(self.holdout_y)), np.array(self.holdout_y_hat)],
                  'train': [one_hot(np.array(self.train_y)), np.array(self.train_y_hat)]}

        for k, y_s in to_run.items():
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(2):
                fpr[i], tpr[i], _ = roc_curve(y_s[0][:, i], y_s[1][:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_s[0].ravel(), y_s[1].ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            plt.figure()
            lw = 2
            plt.plot(fpr[1], tpr[1], color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(self.config.data_dir_save, f'roc_{k}.png'))
            self.performance_data[f"auc_{k}"] = roc_auc[1]

    def calc_precision_recall(self):
        to_run = {'train': [self.train_y, self.get_pos_values(self.train_y_hat)],
                  'holdout': [self.holdout_y, self.get_pos_values(self.holdout_y_hat)]}
        for k, y_s in to_run.items():

            self.performance_data[f'p_r_curve_{k}'] = self.create_p_r_dict(precision_recall_curve(y_s[0],
                                                                                                  y_s[1]))

            self.performance_data[f'avg_precision_micro_{k}'] = average_precision_score(y_s[0], y_s[1], average="micro")
            plt.figure()
            lw = 2
            plt.plot(self.performance_data[f'p_r_curve_{k}']['recall'],
                     self.performance_data[f'p_r_curve_{k}']['precision'],
                     color='darkorange',
                     lw=lw, label='Average precision score, micro-averaged over all classes: {0:0.2f}'
                     .format(self.performance_data[f'avg_precision_micro_{k}']))
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision Recall Curve')
            plt.legend(loc="lower right")

            plt.savefig(os.path.join(self.config.data_dir_save, f'precision_recall_{k}.png'))

    @staticmethod
    def create_p_r_dict(prc):
        o = dict()
        o['precision'] = prc[0].tolist()
        o['recall'] = prc[1].tolist()
        o['threshold'] = prc[2].tolist()

        return o

    def save_performance_data(self):
        o = json.dumps(self.performance_data)
        f = open(self.config.data_dir_save + "/performance.json", "w+")
        f.write(o)
        f.close()

    @staticmethod
    def get_pos_values(binary_outcome):
        return [i[1] for i in binary_outcome]

    def save_predictions(self, dir):
        train_file = os.path.join(dir, str(self.review_catalogue.uuid) + '_' + 'train_predictions.csv')
        holdout_file = os.path.join(dir, str(self.review_catalogue.uuid) + '_' + 'holdout_predictions.csv')

        with open(train_file, 'w+') as f:
            f.write('y,y_hat\n')
            for i in range(len(self.train_y_hat)):
                f.write(str(self.train_y[i]) + ',' + str(self.train_y_hat[i][1]) + '\n')

        with open(holdout_file, 'w+') as f:
            f.write('y,y_hat\n')
            for i in range(len(self.holdout_y_hat)):
                f.write(str(self.holdout_y[i]) + ',' + str(self.holdout_y_hat[i][1]) + '\n')
