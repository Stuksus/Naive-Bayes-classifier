# coding: utf-8
import numpy as np
from scipy.special import logsumexp


class GaussianDistribution:
    def __init__(self, feature):
        self.mean = feature.mean(axis=0)
        self.std = feature.std(axis=0)

    def logpdf(self, value):
        return np.log(self.pdf(value))

    def pdf(self, value):
        return (self.std * (2 * np.pi) ** 0.5) ** (-1) * np.exp(-(value - self.mean) ** 2 / (2 * self.std ** 2))


class NaiveBayes:
    def __init__(self):
        self.target_proba = {}
        self.label_distribution = {}
        self.unique_labels = np.array([])
        self.distribution = []
        self.start_message = '''
        Приветствую! Это реализация наивного Байеса, написанная в учебных целях.
        Автор:
            - GitHub: https://github.com/Stuksus
            - E-mail: https://vk.com/antonsmetanin
            - VK: aasmetanin@edu.hse.ru
        '''
        print(self.start_message)

    def fit(self, design_matrix, target, distribution=None):
        if distribution is None:
            self.distribution = [GaussianDistribution] * design_matrix.shape[1]
        else:
            assert len(distribution) == design_matrix.shape[1]
            self.distribution = distribution

        self.unique_labels = np.unique(target)
        self.label_distribution = {}
        for label in self.unique_labels:
            pretrain_distribution = []
            for idx in range(design_matrix.shape[1]):
                data = design_matrix[target == label, idx]
                distributon_idx = self.distribution[idx](data)
                pretrain_distribution.append(distributon_idx)
            self.label_distribution[label] = pretrain_distribution

        self.target_proba = {
            label: sum(target == label).astype('float') / len(target)
            for label in self.unique_labels
        }

    def prediction_log_proba(self, test):
        log_probas = np.zeros((test.shape[0], len(self.unique_labels)), dtype=float)
        for label_idx, label in enumerate(self.unique_labels):
            for idx in range(test.shape[1]):
                log_probas[:, label_idx] += self.label_distribution[label][idx].logpdf(test[:, idx])
            log_probas[:, label_idx] += np.log(self.target_proba[label])

        for idx in range(len(log_probas)):
            log_probas -= logsumexp(log_probas, axis=1)[:, None]
        return log_probas

    def proba(self, test):
        probas = self.prediction_log_proba(test)
        return np.exp(probas)

    def prediction(self, test):
        prediction_proba = self.prediction_log_proba(test)
        prediction_target = [self.unique_labels[idx] for idx in prediction_proba.argmax(axis=1)]
        return prediction_target
