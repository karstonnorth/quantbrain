import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler

class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class PricePredictor:
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.model = LSTMPredictor()
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, data: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for LSTM model.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (X, y) tensors
        """
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length])
            
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def train(
        self,
        data: pd.Series,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.01
    ) -> List[float]:
        """
        Train the LSTM model.
        
        Args:
            data: Training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            List of training losses
        """
        X, y = self.prepare_data(data)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        losses = []
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(X) / batch_size)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
                
        return losses
    
    def predict(self, data: pd.Series, steps: int = 5) -> pd.Series:
        """
        Make predictions using the trained model.
        
        Args:
            data: Input data for prediction
            steps: Number of steps to predict
            
        Returns:
            Series of predictions
        """
        self.model.eval()
        with torch.no_grad():
            scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
            last_sequence = torch.FloatTensor(scaled_data[-self.sequence_length:]).unsqueeze(0)
            
            predictions = []
            for _ in range(steps):
                pred = self.model(last_sequence)
                predictions.append(pred.item())
                last_sequence = torch.cat([
                    last_sequence[:, 1:, :],
                    pred.unsqueeze(0).unsqueeze(0)
                ], dim=1)
            
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            return pd.Series(predictions.flatten()) 