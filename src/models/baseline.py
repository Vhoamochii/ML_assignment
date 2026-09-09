"""Baseline models for text classification"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import numpy as np


class BaselineModels:
    """Simple baseline models"""
    
    @staticmethod
    def get_dummy_classifier(strategy='most_frequent'):
        """Get dummy classifier (baseline)"""
        return DummyClassifier(strategy=strategy, random_state=42)
    
    @staticmethod
    def get_naive_bayes(alpha=1.0):
        """Get Naive Bayes classifier"""
        return MultinomialNB(alpha=alpha)
    
    @staticmethod
    def get_logistic_regression(max_iter=1000, C=1.0):
        """Get Logistic Regression classifier"""
        return LogisticRegression(
            max_iter=max_iter,
            C=C,
            random_state=42,
            n_jobs=-1
        )


if __name__ == '__main__':
    # Test model creation
    nb_model = BaselineModels.get_naive_bayes()
    lr_model = BaselineModels.get_logistic_regression()
    print("Baseline models created successfully")
