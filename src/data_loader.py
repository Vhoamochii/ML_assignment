"""Data loading and splitting utilities for AG News dataset"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
import os


class AGNewsLoader:
    """Load and prepare AG News dataset"""
    
    def __init__(self, data_path: str = 'data/raw'):
        self.data_path = data_path
        self.class_names = {
            1: 'World',
            2: 'Sports', 
            3: 'Business',
            4: 'Sci/Tech'
        }
    
    def load_data(self, train_file: str = 'train.csv', test_file: str = 'test.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test CSV files"""
        train_path = os.path.join(self.data_path, train_file)
        test_path = os.path.join(self.data_path, test_file)
        
        # AG News format: class_index, title, description
        columns = ['class', 'title', 'description']
        
        train_df = pd.read_csv(train_path, names=columns)
        test_df = pd.read_csv(test_path, names=columns)
        
        return train_df, test_df
    
    def combine_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine title and description into single text field"""
        df['text'] = df['title'] + ' ' + df['description']
        return df
    
    def add_class_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add human-readable class names"""
        df['class_name'] = df['class'].map(self.class_names)
        return df
    
    def prepare_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare full dataset"""
        train_df, test_df = self.load_data()
        
        train_df = self.combine_text(train_df)
        train_df = self.add_class_names(train_df)
        
        test_df = self.combine_text(test_df)
        test_df = self.add_class_names(test_df)
        
        return train_df, test_df
    
    def create_validation_split(self, train_df: pd.DataFrame, val_size: float = 0.15, 
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split training data into train and validation sets"""
        train_data, val_data = train_test_split(
            train_df, 
            test_size=val_size,
            random_state=random_state,
            stratify=train_df['class']
        )
        
        return train_data, val_data


if __name__ == '__main__':
    loader = AGNewsLoader()
    train_df, test_df = loader.prepare_dataset()
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"\nClass distribution:\n{train_df['class_name'].value_counts()}")
