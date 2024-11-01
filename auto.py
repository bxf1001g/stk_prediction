import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from binance.client import Client
from binance.enums import *
import ta
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import smtplib
from email.message import EmailMessage
from scipy import stats

# Set up your API key and secret
api_key = 'G0ph7pTOtOWf98iy1ZQPvCAQgidoXPi3HkXGcXQqT1V2nQlRVbeC6fBZXEVs3skP'
api_secret = 'HTqEyhSPe0dFAopYiLP7YJG8XpLJIXjvzwIUZ0MqW25xeslQgmHl5MXt1bxPAc6m'


# Initialize the Binance client
client = Client(api_key, api_secret)

# Define your symbol and interval
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1MINUTE
seq_length = 60  # Sequence length for model
start_bal = 10000

# Feature columns
feature_cols = ['close', 'volume', 'ma5', 'ma10', 'ma20', 'rsi',
                'macd', 'macd_diff', 'macd_signal', 'bb_h', 'bb_l',
                'open', 'high', 'low','ema50', 'stochastic_k']

# Global variables for normalization
data_mean = 0
data_std = 1
target_mean = 0
target_std = 1

def get_historical_data(symbol, interval, start_str, end_str=None):
    klines = client.get_historical_klines(
        symbol, interval, start_str, end_str)
    data = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
    data.set_index('timestamp', inplace=True)
    data = data.astype(float)
    data = data.dropna()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    z_scores = np.abs(stats.zscore(data[['open', 'high', 'low', 'close', 'volume']]))
    data = data[(z_scores < 3).all(axis=1)]
    return data

def add_technical_indicators(data):
    data['ma5'] = data['close'].rolling(window=5).mean()
    data['ema50'] = data['close'].ewm(span=50, adjust=False).mean()
    data['stochastic_k'] = ta.momentum.StochasticOscillator(
    data['high'], data['low'], data['close'], window=14).stoch()
    data['ma10'] = data['close'].rolling(window=10).mean()
    data['ma20'] = data['close'].rolling(window=20).mean()
    data['rsi'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
    macd = ta.trend.MACD(data['close'])
    data['macd'] = macd.macd()
    data['macd_diff'] = macd.macd_diff()
    data['macd_signal'] = macd.macd_signal()
    bollinger = ta.volatility.BollingerBands(data['close'], window=20)
    data['bb_h'] = bollinger.bollinger_hband()
    data['bb_l'] = bollinger.bollinger_lband()
    data.bfill(inplace=True)
    data.ffill(inplace=True)
    return data

def prepare_data(data):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length][feature_cols].values
        target = data.iloc[i+seq_length]['close']
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

class HybridCNNLSTMModel(nn.Module):
    def __init__(self, input_size, cnn_out_channels, lstm_hidden_size,
                 lstm_num_layers, output_size, dropout_rate):
        super(HybridCNNLSTMModel, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_size,
                             out_channels=cnn_out_channels,
                             kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_out_channels, lstm_hidden_size,
                            lstm_num_layers, batch_first=True,
                            dropout=dropout_rate)
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_length)
        x = self.relu(self.cnn(x))
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, cnn_out_channels)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0),
                         self.lstm.hidden_size)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0),
                         self.lstm.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_model():
    global data_mean, data_std, target_mean, target_std
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=30)
    start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
    historical_data = get_historical_data(symbol, interval, start_str, end_str)
    data = add_technical_indicators(historical_data)
    data = data.reset_index()
    sequences, targets = prepare_data(data)
    train_size = int(len(sequences) * 0.8)
    X_train = sequences[:train_size]
    y_train = targets[:train_size]
    X_val = sequences[train_size:]
    y_val = targets[train_size:]
    data_mean = X_train.mean()
    data_std = X_train.std()
    target_mean = y_train.mean()
    target_std = y_train.std()
    np.save('data_mean.npy', data_mean)
    np.save('data_std.npy', data_std)
    np.save('target_mean.npy', target_mean)
    np.save('target_std.npy', target_std)
    X_train = (X_train - data_mean) / data_std
    X_val = (X_val - data_mean) / data_std
    y_train = (y_train - target_mean) / target_std
    y_val = (y_val - target_mean) / target_std
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    param_grid = {
        'learning_rate': [0.0005],
        'batch_size': [128],
        'num_epochs': [30]
    }
    best_val_loss = float('inf')
    for params in ParameterGrid(param_grid):
        learning_rate = params['learning_rate']
        batch_size = params['batch_size']
        num_epochs = params['num_epochs']
        model = HybridCNNLSTMModel(
            input_size=len(feature_cols),
            cnn_out_channels=64,
            lstm_hidden_size=128,
            lstm_num_layers=2,
            output_size=1,
            dropout_rate=0.3)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    val_loss = criterion(outputs, y_batch)
                    val_losses.append(val_loss.item())
            avg_val_loss = np.mean(val_losses)
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    print('Training completed.')

def trading():
    model = HybridCNNLSTMModel(
        input_size=len(feature_cols),
        cnn_out_channels=64,
        lstm_hidden_size=128,
        lstm_num_layers=2,
        output_size=1,
        dropout_rate=0.3)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    data_mean = np.load('data_mean.npy')
    data_std = np.load('data_std.npy')
    target_mean = np.load('target_mean.npy')
    target_std = np.load('target_std.npy')
    initial_balance = 10000
    balance = initial_balance
    position = 0
    transaction_fee = 0.001  # 0.1% transaction fee
    cumulative_reward = 0
    stop_loss_pct = 0.98
    take_profit_pct = 1.02
    threshold_buy = 0.2
    threshold_sell = -0.2

    while True:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=1)
        start_str = start_time.strftime('%d %b %Y %H:%M:%S')
        end_str = end_time.strftime('%d %b %Y %H:%M:%S')
        historical_data = get_historical_data(symbol, interval, start_str, end_str)
        data = add_technical_indicators(historical_data)
        data = data[-(seq_length + 1):]
        data.reset_index(drop=True, inplace=True)
        sequence = data.iloc[:-1][feature_cols].values
        sequence = (sequence - data_mean) / data_std
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prediction_normalized = model(sequence_tensor).item()
        prediction = (prediction_normalized * target_std) + target_mean
        current_price = data.iloc[-1]['close']
        price_diff_pct = ((prediction - current_price) / current_price) * 100
        current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        print(f"Time: {current_time}, Predicted Price: {prediction:.4f}, Current Price: {current_price:.4f}")
        print(f"Price Difference: {price_diff_pct:.2f}%")

        if position == 0:
            if price_diff_pct >= threshold_buy:
                # Calculate the amount of cryptocurrency to buy, considering the transaction fee
                position = (balance / current_price) * (1 - transaction_fee)
                buy_price = current_price
                balance = 0
                print(f"Buying at {current_price:.4f}")
        else:
            if (price_diff_pct <= threshold_sell or
                current_price <= buy_price * stop_loss_pct or
                current_price >= buy_price * take_profit_pct):
                # Calculate the new balance after selling, considering the transaction fee
                sell_price = current_price
                new_balance = position * sell_price * (1 - transaction_fee)
                reward = new_balance - initial_balance
                cumulative_reward += reward
                balance = new_balance
                position = 0
                print(f"Selling at {sell_price:.4f}, Reward: {reward:.2f}")

        if position > 0:
            total_assets = balance + position * current_price * (1 - transaction_fee)
        else:
            total_assets = balance
        print(f"Current balance: ${total_assets:.2f}, Cumulative Reward: {cumulative_reward:.2f}")

        # Add this line to display position, balance, and price difference percentage
        print(f"Position: {position}, Balance: {balance}, Price Diff %: {price_diff_pct:.2f}%")

        time.sleep(1)  # Check every second

def main():
    if not os.path.exists('/content/best_model.pth'):
        train_model()
    trading()

if __name__ == "__main__":
    main()
