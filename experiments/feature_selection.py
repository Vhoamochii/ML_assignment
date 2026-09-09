"""Feature selection experiments with Bag of Words and TF-IDF"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import (
    SelectKBest, chi2, mutual_info_classif,
    SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class FeatureRepresentation:
    """Create feature representations: Bag of Words (BoW) and TF-IDF"""
    
    def __init__(self, method='tfidf', max_features=5000, ngram_range=(1, 1)):
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        
        if method == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2
            )
        elif method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'bow' or 'tfidf'")
    
    def fit_transform(self, texts):
        """Fit vectorizer and transform texts"""
        X = self.vectorizer.fit_transform(texts)
        print(f"{self.method.upper()} - Feature matrix shape: {X.shape}")
        return X
    
    def transform(self, texts):
        """Transform texts using fitted vectorizer"""
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names (vocabulary)"""
        return self.vectorizer.get_feature_names_out()


class FeatureSelector:
    """Feature selection methods for text classification"""
    
    def __init__(self, method='chi2', k=1000):
        self.method = method
        self.k = k
        self.selector = None
        self.selected_features_mask = None
    
    def fit_transform(self, X_train, y_train):
        """Fit and transform training data"""
        if self.method == 'chi2':
            self.selector = SelectKBest(chi2, k=self.k)
        elif self.method == 'mutual_info':
            self.selector = SelectKBest(mutual_info_classif, k=self.k)
        elif self.method == 'tree_based':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            self.selector = SelectFromModel(estimator, max_features=self.k)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        X_train_selected = self.selector.fit_transform(X_train, y_train)
        self.selected_features_mask = self.selector.get_support()
        
        print(f"{self.method.upper()} - Selected {X_train_selected.shape[1]} features from {X_train.shape[1]}")
        return X_train_selected
    
    def transform(self, X_test):
        """Transform test data"""
        if self.selector is None:
            raise ValueError("Selector not fitted. Call fit_transform first.")
        return self.selector.transform(X_test)
    
    def get_feature_scores(self):
        """Get feature importance scores"""
        if hasattr(self.selector, 'scores_'):
            return self.selector.scores_
        return None
    
    def get_selected_feature_names(self, all_feature_names):
        """Get names of selected features"""
        if self.selected_features_mask is None:
            return None
        return all_feature_names[self.selected_features_mask]
    
    def plot_feature_importance(self, feature_names, top_n=20, save_path=None):
        """Plot top features by importance score"""
        scores = self.get_feature_scores()
        
        if scores is None:
            print("Feature scores not available for this method")
            return
        
        # Get top features
        top_indices = np.argsort(scores)[-top_n:]
        top_scores = scores[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), top_scores)
        plt.yticks(range(top_n), top_names)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Features ({self.method.upper()})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_feature_representations(self, scores_dict, top_n=15, save_path=None):
        fig, axes = plt.subplots(1, len(scores_dict), figsize=(15, 8))
        
        if len(scores_dict) == 1:
            axes = [axes]
        
        for idx, (rep_name, (feature_names, scores)) in enumerate(scores_dict.items()):
            top_indices = np.argsort(scores)[-top_n:]
            top_scores = scores[top_indices]
            top_names = [feature_names[i] for i in top_indices]
            
            axes[idx].barh(range(top_n), top_scores)
            axes[idx].set_yticks(range(top_n))
            axes[idx].set_yticklabels(top_names)
            axes[idx].set_xlabel('Score')
            axes[idx].set_title(f'{rep_name}\nTop {top_n} Features')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def compare_bow_vs_tfidf(texts_train, texts_test, y_train, y_test, 
                         max_features=5000, selection_method='chi2', k_best=1000):
    """
    Compare Bag of Words vs TF-IDF as feature representations
    
    Args:
        texts_train: Training text data
        texts_test: Test text data
        y_train: Training labels
        y_test: Test labels
        max_features: Maximum features to extract
        selection_method: Feature selection method
        k_best: Number of best features to select
    
    Returns:
        Dictionary with results for both representations
    """
    results = {}
    
    for method in ['bow', 'tfidf']:
        print(f"\n{'='*60}")
        print(f"Processing {method.upper()}")
        print('='*60)
        
        # Create feature representation
        feature_rep = FeatureRepresentation(method=method, max_features=max_features)
        X_train = feature_rep.fit_transform(texts_train)
        X_test = feature_rep.transform(texts_test)
        
        # Feature selection
        selector = FeatureSelector(method=selection_method, k=k_best)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Store results
        results[method] = {
            'X_train': X_train_selected,
            'X_test': X_test_selected,
            'feature_names': feature_rep.get_feature_names(),
            'selector': selector,
            'vectorizer': feature_rep.vectorizer
        }
    
    return results


if __name__ == '__main__':
    print("Feature selection module loaded")
    print("Available representations: 'bow' (Bag of Words), 'tfidf' (TF-IDF)")
    print("Available selection methods: 'chi2', 'mutual_info', 'tree_based'")
