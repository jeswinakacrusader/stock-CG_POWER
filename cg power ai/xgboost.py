import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class EnsembleModel:
    def __init__(self, input_size, sequence_length):
        self.lstm = LSTM(input_size, hidden_size=64, num_layers=2, output_size=1)
        self.rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.xgb = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
        self.scaler = MinMaxScaler()
        self.sequence_length = sequence_length

    def prepare_data(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_sequences = []
        y_sequences = []
        for i in range(len(X_scaled) - self.sequence_length):
            X_sequences.append(X_scaled[i:i+self.sequence_length])
            y_sequences.append(y[i+self.sequence_length])
        return np.array(X_sequences), np.array(y_sequences)

    def train(self, X, y):
        X_seq, y_seq = self.prepare_data(X, y)
        X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

        # Train LSTM
        train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train.reshape(-1, 1)))
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm.parameters())
        
        for epoch in range(100):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.lstm(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Train Random Forest
        self.rf.fit(X_train.reshape(X_train.shape[0], -1), y_train)

        # Train XGBoost
        self.xgb.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_seq = np.array([X_scaled[-self.sequence_length:]])
        
        lstm_pred = self.lstm(torch.FloatTensor(X_seq)).detach().numpy()
        rf_pred = self.rf.predict(X_seq.reshape(1, -1))
        xgb_pred = self.xgb.predict(X_seq.reshape(1, -1))
        
        ensemble_pred = (lstm_pred + rf_pred + xgb_pred) / 3
        return ensemble_pred[0][0]

# Usage example:
# ensemble_model = EnsembleModel(input_size=10, sequence_length=10)
# ensemble_model.train(X_train, y_train)
# prediction = ensemble_model.predict(X_test)