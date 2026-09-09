"""Hyperparameter tuning experiments"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
import numpy as np
import sys
sys.path.append('..')


class HyperparameterTuner:
    """Perform hyperparameter tuning"""
    
    def __init__(self, scoring='f1_macro', cv=5, n_jobs=-1):
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
    
    def grid_search(self, model, param_grid, X_train, y_train):
        """Perform grid search"""
        grid_search = GridSearchCV(
            model,
            param_grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def random_search(self, model, param_distributions, X_train, y_train, n_iter=20):
        """Perform randomized search"""
        random_search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=2,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_, random_search.best_params_


# Example parameter grids
PARAM_GRIDS = {
    'logistic_regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'max_iter': [1000, 2000]
    },
    'svm': {
        'C': [0.1, 1, 10],
        'max_iter': [1000, 2000]
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    },
    'naive_bayes': {
        'alpha': [0.1, 0.5, 1.0, 2.0]
    }
}


if __name__ == '__main__':
    print("Hyperparameter tuning module loaded")
    print("Available parameter grids:", list(PARAM_GRIDS.keys()))
