"""Model evaluation utilities"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time


class ModelEvaluator:
    """Evaluate and compare machine learning models"""
    
    def __init__(self, class_names=None):
        self.class_names = class_names or ['World', 'Sports', 'Business', 'Sci/Tech']
    
    def evaluate_model(self, y_true, y_pred, model_name='Model'):
        """Calculate evaluation metrics"""
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        }
        
        return metrics
    
    def print_classification_report(self, y_true, y_pred):
        """Print detailed classification report"""
        print(classification_report(y_true, y_pred, target_names=self.class_names))
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self, results_list):
        """Compare multiple models"""
        df = pd.DataFrame(results_list)
        df = df.round(4)
        return df
    
    def plot_model_comparison(self, results_df, save_path=None):
        """Plot model comparison"""
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            axes[idx].bar(results_df['model_name'], results_df[metric])
            axes[idx].set_title(f'{metric.replace("_", " ").title()}')
            axes[idx].set_ylabel('Score')
            axes[idx].set_ylim(0, 1)
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_and_evaluate(self, model, X_train, y_train, X_test, y_test, model_name='Model'):
        """Train model and return evaluation metrics with training time"""
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time
        
        # Evaluate
        metrics = self.evaluate_model(y_test, y_pred, model_name)
        metrics['train_time'] = train_time
        metrics['predict_time'] = predict_time
        
        return metrics, y_pred


if __name__ == '__main__':
    # Test evaluator
    evaluator = ModelEvaluator()
    print("Evaluator initialized with classes:", evaluator.class_names)
