"""Utility functions"""

import joblib
import json
import os
from datetime import datetime


def save_model(model, filepath):
    """Save model to disk"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load model from disk"""
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def save_metrics(metrics, filepath):
    """Save metrics to JSON"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filepath}")


def load_metrics(filepath):
    """Load metrics from JSON"""
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    return metrics


def create_experiment_folder(base_path='results'):
    """Create timestamped experiment folder"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_path = os.path.join(base_path, f'experiment_{timestamp}')
    os.makedirs(exp_path, exist_ok=True)
    return exp_path


def ensure_directories():
    """Create necessary project directories"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'results/figures',
        'results/metrics'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Project directories created")


if __name__ == '__main__':
    ensure_directories()
