"""Neural network models (optional)"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TextClassificationDataset(Dataset):
    """PyTorch dataset for text classification"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.toarray() if hasattr(X, 'toarray') else X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleNN(nn.Module):
    """Simple feedforward neural network"""
    
    def __init__(self, input_size, hidden_size=128, num_classes=4, dropout=0.5):
        super(SimpleNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class NeuralNetworkClassifier:
    """Wrapper for neural network classifier"""
    
    def __init__(self, input_size, hidden_size=128, num_classes=4, 
                 learning_rate=0.001, epochs=10, batch_size=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleNN(input_size, hidden_size, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size
    
    def fit(self, X_train, y_train):
        """Train the model"""
        # Adjust labels to 0-indexed
        y_train_adjusted = y_train - 1 if y_train.min() == 1 else y_train
        
        dataset = TextClassificationDataset(X_train, y_train_adjusted)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, X_test):
        """Make predictions"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X_test.toarray() if hasattr(X_test, 'toarray') else X_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # Adjust back to 1-indexed if needed
        return predictions + 1
    
    def predict_proba(self, X_test):
        """Predict probabilities"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X_test.toarray() if hasattr(X_test, 'toarray') else X_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return probabilities


if __name__ == '__main__':
    print("Neural network models module loaded")
    print(f"CUDA available: {torch.cuda.is_available()}")
