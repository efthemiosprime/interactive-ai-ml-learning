import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function TradingToolsTutorial() {
  const [selectedStep, setSelectedStep] = useState('step1');
  const [selectedFramework, setSelectedFramework] = useState('pytorch');

  const steps = {
    step1: {
      title: 'Step 1: Data Collection & Preprocessing',
      description: 'Gather financial market data and prepare it for analysis',
      code: `# Step 1: Data Collection & Preprocessing
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# 1.1 Collect Stock Market Data
def fetch_stock_data(symbol, period='1y'):
    """
    Fetch historical stock data using yfinance
    
    Parameters:
    - symbol: Stock ticker (e.g., 'AAPL', 'GOOGL', 'MSFT')
    - period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)
    return data

# Example: Fetch Apple stock data
aapl_data = fetch_stock_data('AAPL', period='2y')
print(f"Fetched {len(aapl_data)} days of data")
print(aapl_data.head())

# 1.2 Data Structure
# Each row contains:
# - Open: Opening price
# - High: Highest price of the day
# - Low: Lowest price of the day
# - Close: Closing price (most important for prediction)
# - Volume: Number of shares traded

# 1.3 Handle Missing Data
def clean_data(df):
    """Remove missing values and outliers"""
    # Forward fill missing values
    df = df.fillna(method='ffill')
    
    # Backward fill any remaining NaN
    df = df.fillna(method='bfill')
    
    # Remove outliers (prices that are 3 standard deviations away)
    for column in ['Open', 'High', 'Low', 'Close']:
        mean = df[column].mean()
        std = df[column].std()
        df = df[(df[column] >= mean - 3*std) & (df[column] <= mean + 3*std)]
    
    return df

cleaned_data = clean_data(aapl_data.copy())

# 1.4 Feature Engineering: Basic Features
def add_basic_features(df):
    """Add basic derived features"""
    df = df.copy()
    
    # Price change (day-to-day)
    df['Price_Change'] = df['Close'].diff()
    
    # Percentage change
    df['Pct_Change'] = df['Close'].pct_change()
    
    # High-Low spread
    df['HL_Spread'] = df['High'] - df['Low']
    
    # Volume-weighted average price (VWAP)
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    return df

featured_data = add_basic_features(cleaned_data)
print(featured_data[['Close', 'Price_Change', 'Pct_Change', 'HL_Spread']].head())

# 1.5 Normalize Data
# Important for neural networks - scale features to similar ranges
from sklearn.preprocessing import MinMaxScaler

def normalize_features(df, feature_columns):
    """Normalize features to [0, 1] range"""
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df, scaler

# Select features to normalize
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'Price_Change', 'Pct_Change', 'HL_Spread', 'VWAP']
normalized_data, scaler = normalize_features(featured_data.copy(), feature_cols)

print("Data preprocessing complete!")
print(f"Shape: {normalized_data.shape}")
print(f"Features: {feature_cols}")`
    },
    step2: {
      title: 'Step 2: Mathematical Foundations - Technical Indicators',
      description: 'Calculate technical indicators using mathematical formulas',
      code: `# Step 2: Mathematical Foundations - Technical Indicators
import pandas as pd
import numpy as np

# 2.1 Moving Averages
# Moving averages smooth out price fluctuations to identify trends

def calculate_sma(data, window):
    """
    Simple Moving Average (SMA)
    Formula: SMA(n) = (P₁ + P₂ + ... + Pₙ) / n
    where Pᵢ = price at time i, n = window size
    """
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    """
    Exponential Moving Average (EMA)
    Formula: EMA(t) = α · P(t) + (1-α) · EMA(t-1)
    where α = 2/(n+1), n = window size, P(t) = current price
    """
    return data.ewm(span=window, adjust=False).mean()

# Example calculation:
# Prices: [100, 102, 101, 103, 105]
# SMA(3) at day 5: (101 + 103 + 105) / 3 = 103
# EMA(3) uses exponential weighting, giving more weight to recent prices

# 2.2 Relative Strength Index (RSI)
# Measures momentum - indicates overbought (>70) or oversold (<30) conditions

def calculate_rsi(data, window=14):
    """
    Relative Strength Index (RSI)
    Formula: RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over n periods
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Example calculation:
# If average gain = 2, average loss = 1 over 14 days
# RS = 2/1 = 2
# RSI = 100 - (100 / (1 + 2)) = 100 - 33.33 = 66.67

# 2.3 MACD (Moving Average Convergence Divergence)
# Shows relationship between two EMAs

def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    MACD = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD, signal_period)
    Histogram = MACD - Signal Line
    """
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

# 2.4 Bollinger Bands
# Volatility indicator - bands expand/contract based on volatility

def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    Bollinger Bands
    Middle Band = SMA(window)
    Upper Band = SMA + (num_std × Standard Deviation)
    Lower Band = SMA - (num_std × Standard Deviation)
    """
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    
    return upper_band, sma, lower_band

# Example calculation:
# If SMA(20) = 100, Std = 5, num_std = 2
# Upper Band = 100 + (2 × 5) = 110
# Lower Band = 100 - (2 × 5) = 90

# 2.5 Stochastic Oscillator
# Compares closing price to price range over a period

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    """
    Stochastic Oscillator
    %K = 100 × (C - L14) / (H14 - L14)
    %D = SMA(%K, d_window)
    where C = current close, L14 = lowest low in 14 periods, H14 = highest high
    """
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = calculate_sma(k_percent, d_window)
    
    return k_percent, d_percent

# 2.6 Apply All Indicators
def add_technical_indicators(df):
    """Add all technical indicators to dataframe"""
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Moving Averages
    df['SMA_20'] = calculate_sma(close, 20)
    df['SMA_50'] = calculate_sma(close, 50)
    df['EMA_12'] = calculate_ema(close, 12)
    df['EMA_26'] = calculate_ema(close, 26)
    
    # RSI
    df['RSI'] = calculate_rsi(close, 14)
    
    # MACD
    macd, signal, histogram = calculate_macd(close)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Histogram'] = histogram
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
    df['BB_Upper'] = bb_upper
    df['BB_Middle'] = bb_middle
    df['BB_Lower'] = bb_lower
    
    # Stochastic
    stoch_k, stoch_d = calculate_stochastic(high, low, close)
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_d
    
    return df

# Example usage
# df_with_indicators = add_technical_indicators(normalized_data)
# print(df_with_indicators[['Close', 'SMA_20', 'RSI', 'MACD']].tail())`
    },
    step3: {
      title: 'Step 3: Feature Engineering & Data Preparation',
      description: 'Create sequences and prepare data for ML models',
      code: `# Step 3: Feature Engineering & Data Preparation
import numpy as np
from sklearn.model_selection import train_test_split

# 3.1 Create Sequences for Time Series Prediction
# LSTM/RNN models need sequences of historical data to predict future values

def create_sequences(data, sequence_length, prediction_horizon=1):
    """
    Create sequences for time series prediction
    
    Parameters:
    - data: DataFrame with features
    - sequence_length: Number of time steps to look back (e.g., 60 days)
    - prediction_horizon: How many steps ahead to predict (default: 1 day)
    
    Returns:
    - X: Input sequences [samples, sequence_length, features]
    - y: Target values [samples, prediction_horizon]
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        # Input: sequence_length days of data
        X.append(data[i:i+sequence_length].values)
        
        # Target: prediction_horizon days ahead
        y.append(data[i+sequence_length:i+sequence_length+prediction_horizon]['Close'].values)
    
    return np.array(X), np.array(y)

# Example:
# If sequence_length = 5, prediction_horizon = 1
# X[0] = data[0:5] (days 0-4)
# y[0] = data[5]['Close'] (day 5 close price)

# 3.2 Select Features for Model
def prepare_features(df):
    """Select and prepare features for model training"""
    # Use technical indicators and basic features
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Upper', 'BB_Middle', 'BB_Lower',
        'Stoch_K', 'Stoch_D'
    ]
    
    # Remove NaN values (from indicator calculations)
    df_clean = df[feature_columns].dropna()
    
    return df_clean, feature_columns

# 3.3 Split Data into Train/Validation/Test Sets
def split_time_series_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Split time series data chronologically (not randomly!)
    Important: Don't shuffle time series data
    """
    n_samples = len(X)
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# 3.4 Complete Data Preparation Pipeline
def prepare_training_data(df, sequence_length=60, prediction_horizon=1):
    """
    Complete pipeline to prepare data for training
    """
    # Add technical indicators
    df_with_indicators = add_technical_indicators(df)
    
    # Prepare features
    df_features, feature_cols = prepare_features(df_with_indicators)
    
    # Create sequences
    X, y = create_sequences(df_features, sequence_length, prediction_horizon)
    
    # Split data
    train_data, val_data, test_data = split_time_series_data(X, y)
    
    print(f"Training samples: {len(train_data[0])}")
    print(f"Validation samples: {len(val_data[0])}")
    print(f"Test samples: {len(test_data[0])}")
    print(f"Sequence length: {sequence_length}")
    print(f"Features per timestep: {len(feature_cols)}")
    
    return train_data, val_data, test_data, feature_cols

# Example usage:
# train, val, test, features = prepare_training_data(normalized_data, sequence_length=60)
# X_train, y_train = train
# X_val, y_val = val
# X_test, y_test = test`
    },
    step4: {
      title: 'Step 4: LSTM Model for Price Prediction',
      description: 'Build and train LSTM neural network for stock price prediction',
      code: `# Step 4: LSTM Model for Price Prediction
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 4.1 LSTM Architecture
# LSTM (Long Short-Term Memory) is perfect for time series prediction
# It can remember patterns over long sequences

class StockPriceLSTM(nn.Module):
    """
    LSTM Model for Stock Price Prediction
    
    Architecture:
    - Input Layer: Takes sequences of features
    - LSTM Layers: Process temporal patterns
    - Dense Layers: Output prediction
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(StockPriceLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True  # (batch, seq, features)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)  # Output: single price prediction
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass through LSTM
        
        x shape: (batch_size, sequence_length, input_size)
        """
        # LSTM output: (batch, seq, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last timestep's output
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Fully connected layers
        x = self.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, 1)
        
        return x

# 4.2 Training Function
def train_lstm_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001):
    """
    Train LSTM model
    
    Loss Function: Mean Squared Error (MSE)
    Optimizer: Adam (adaptive learning rate)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss function: MSE for regression
    criterion = nn.MSELoss()
    
    # Optimizer: Adam with learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses

# 4.3 Create Data Loaders
def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """Create PyTorch data loaders"""
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# 4.4 Complete Training Pipeline
def train_price_prediction_model(X_train, y_train, X_val, y_val, 
                                 input_size, hidden_size=64, num_layers=2):
    """
    Complete pipeline to train price prediction model
    """
    # Create model
    model = StockPriceLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val)
    
    # Train model
    print("Training LSTM model...")
    train_losses, val_losses = train_lstm_model(model, train_loader, val_loader)
    
    return model, train_losses, val_losses

# Example usage:
# model, train_losses, val_losses = train_price_prediction_model(
#     X_train, y_train, X_val, y_val,
#     input_size=len(feature_cols),
#     hidden_size=64,
#     num_layers=2
# )`
    },
    step5: {
      title: 'Step 5: Reinforcement Learning Trading Strategy',
      description: 'Build RL agent to learn optimal trading strategies',
      code: `# Step 5: Reinforcement Learning Trading Strategy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# 5.1 Trading Environment
# RL agent interacts with market environment to learn trading strategies

class TradingEnvironment:
    """
    Simulated trading environment for RL agent
    
    State: Current market features (price, indicators, portfolio)
    Action: Buy, Sell, or Hold
    Reward: Profit/loss from trading actions
    """
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost  # 0.1% fee per trade
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = 0
        self.total_value = self.initial_balance
        self.trades = []
        
        return self.get_state()
    
    def get_state(self):
        """Get current state (market features + portfolio info)"""
        if self.current_step >= len(self.data):
            return None
        
        # Market features
        market_state = self.data[self.current_step]
        
        # Portfolio features
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.shares / 100,  # Normalized shares (assuming max 100 shares)
            self.total_value / self.initial_balance  # Normalized total value
        ])
        
        return np.concatenate([market_state, portfolio_state])
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done)
        
        Actions:
        0: Hold
        1: Buy
        2: Sell
        """
        if self.current_step >= len(self.data) - 1:
            return None, 0, True
        
        current_price = self.data[self.current_step]['Close']
        next_price = self.data[self.current_step + 1]['Close']
        
        reward = 0
        done = False
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > current_price:
                shares_to_buy = int(self.balance / (current_price * (1 + self.transaction_cost)))
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                
                if cost <= self.balance:
                    self.shares += shares_to_buy
                    self.balance -= cost
        
        elif action == 2:  # Sell
            if self.shares > 0:
                revenue = self.shares * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.shares = 0
        
        # Calculate reward (change in portfolio value)
        previous_value = self.total_value
        self.total_value = self.balance + self.shares * next_price
        reward = (self.total_value - previous_value) / self.initial_balance
        
        self.current_step += 1
        
        if self.current_step >= len(self.data) - 1:
            done = True
        
        next_state = self.get_state()
        return next_state, reward, done
    
    def get_portfolio_value(self):
        """Get current total portfolio value"""
        if self.current_step < len(self.data):
            current_price = self.data[self.current_step]['Close']
            return self.balance + self.shares * current_price
        return self.total_value

# 5.2 DQN (Deep Q-Network) Agent
class DQNAgent:
    """
    Deep Q-Network for trading
    
    Q-Learning: Learn optimal action-value function Q(s, a)
    Q(s, a) = Expected future reward from state s, taking action a
    """
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        
        # Q-Network
        self.q_network = self.build_model()
        self.target_network = self.build_model()
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
    
    def build_model(self):
        """Build Q-Network"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        
        Epsilon-greedy:
        - With probability ε: random action (exploration)
        - With probability 1-ε: best action (exploitation)
        """
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self, batch_size=32, gamma=0.95):
        """
        Train Q-Network on batch of experiences
        
        Q-Learning Update:
        Q(s, a) ← Q(s, a) + α[r + γ·max Q(s', a') - Q(s, a)]
        where:
        - α = learning rate
        - γ = discount factor (future reward importance)
        - r = immediate reward
        """
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Current Q values
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (gamma * next_q_values * ~dones)
        
        # Loss: Mean Squared Error
        loss = nn.MSELoss()(q_value, target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, env, episodes=100):
        """Train agent in environment"""
        total_rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            while True:
                action = self.act(state, training=True)
                next_state, reward, done = env.step(action)
                
                if next_state is None:
                    break
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Train on replay buffer
            self.replay()
            
            # Update target network every 10 episodes
            if episode % 10 == 0:
                self.update_target_network()
            
            total_rewards.append(total_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(total_rewards[-10:])
                print(f'Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.4f}, Epsilon: {self.epsilon:.3f}')
        
        return total_rewards

# Example usage:
# env = TradingEnvironment(prepared_data)
# agent = DQNAgent(state_size=len(feature_cols) + 3, action_size=3)
# rewards = agent.train(env, episodes=100)`
    },
    step6: {
      title: 'Step 6: Risk Management & Performance Metrics',
      description: 'Implement risk management strategies and evaluate performance',
      code: `# Step 6: Risk Management & Performance Metrics
import numpy as np
import pandas as pd

# 6.1 Position Sizing
# Risk management: Never risk more than a certain percentage per trade

class RiskManager:
    """
    Risk management for trading strategies
    
    Key Principles:
    1. Position sizing based on risk tolerance
    2. Stop-loss orders to limit losses
    3. Take-profit orders to secure gains
    4. Maximum drawdown limits
    """
    def __init__(self, risk_per_trade=0.02, max_position_size=0.1):
        """
        Parameters:
        - risk_per_trade: Maximum % of portfolio to risk per trade (default: 2%)
        - max_position_size: Maximum % of portfolio in single position (default: 10%)
        """
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
    
    def calculate_position_size(self, portfolio_value, entry_price, stop_loss_price):
        """
        Calculate position size based on risk
        
        Formula:
        Position Size = (Portfolio × Risk%) / |Entry Price - Stop Loss|
        """
        risk_amount = portfolio_value * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0
        
        shares = risk_amount / price_risk
        
        # Limit to max position size
        max_shares = (portfolio_value * self.max_position_size) / entry_price
        shares = min(shares, max_shares)
        
        return int(shares)
    
    def apply_stop_loss(self, current_price, entry_price, stop_loss_pct=0.05):
        """
        Calculate stop-loss price
        
        Stop Loss = Entry Price × (1 - Stop Loss %)
        """
        return entry_price * (1 - stop_loss_pct)
    
    def apply_take_profit(self, current_price, entry_price, take_profit_pct=0.10):
        """
        Calculate take-profit price
        
        Take Profit = Entry Price × (1 + Take Profit %)
        """
        return entry_price * (1 + take_profit_pct)

# 6.2 Performance Metrics
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Sharpe Ratio: Risk-adjusted return
    
    Formula: Sharpe = (Mean Return - Risk-Free Rate) / Std(Returns)
    
    Higher Sharpe = Better risk-adjusted returns
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0
    
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)  # Annualized
    
    return sharpe

def calculate_max_drawdown(portfolio_values):
    """
    Maximum Drawdown: Largest peak-to-trough decline
    
    MDD = (Peak Value - Trough Value) / Peak Value
    """
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)
    
    return max_drawdown

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """
    Sortino Ratio: Like Sharpe, but only penalizes downside volatility
    
    Sortino = (Mean Return - Risk-Free Rate) / Downside Std(Returns)
    """
    if len(returns) == 0:
        return 0
    
    excess_returns = returns - (risk_free_rate / 252)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
    
    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 0
    
    sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
    return sortino

# 6.3 Backtesting Framework
class Backtester:
    """
    Backtesting framework to evaluate trading strategies
    """
    def __init__(self, initial_balance=10000, transaction_cost=0.001):
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reset()
    
    def reset(self):
        self.balance = self.initial_balance
        self.shares = 0
        self.portfolio_values = [self.initial_balance]
        self.trades = []
        self.returns = []
    
    def execute_trade(self, action, price, timestamp):
        """Execute trade and record"""
        if action == 1:  # Buy
            if self.balance > price:
                shares_to_buy = int(self.balance / (price * (1 + self.transaction_cost)))
                cost = shares_to_buy * price * (1 + self.transaction_cost)
                
                if cost <= self.balance:
                    self.shares += shares_to_buy
                    self.balance -= cost
                    self.trades.append({
                        'timestamp': timestamp,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares_to_buy
                    })
        
        elif action == 2:  # Sell
            if self.shares > 0:
                revenue = self.shares * price * (1 - self.transaction_cost)
                self.balance += revenue
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'price': price,
                    'shares': self.shares
                })
                self.shares = 0
    
    def update_portfolio_value(self, current_price):
        """Update portfolio value"""
        total_value = self.balance + self.shares * current_price
        self.portfolio_values.append(total_value)
        
        if len(self.portfolio_values) > 1:
            daily_return = (total_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            self.returns.append(daily_return)
    
    def get_final_value(self, final_price):
        """Get final portfolio value"""
        return self.balance + self.shares * final_price
    
    def generate_report(self, final_price):
        """
        Generate comprehensive performance report
        """
        final_value = self.get_final_value(final_price)
        total_return = (final_value - self.initial_balance) / self.initial_balance
        
        returns_array = np.array(self.returns)
        
        metrics = {
            'Initial Balance': self.initial_balance,
            'Final Value': final_value,
            'Total Return': f"{total_return * 100:.2f}%",
            'Sharpe Ratio': f"{calculate_sharpe_ratio(returns_array):.3f}",
            'Sortino Ratio': f"{calculate_sortino_ratio(returns_array):.3f}",
            'Max Drawdown': f"{calculate_max_drawdown(self.portfolio_values) * 100:.2f}%",
            'Total Trades': len(self.trades),
            'Win Rate': self.calculate_win_rate(),
            'Average Return': f"{np.mean(returns_array) * 100:.2f}%",
            'Volatility': f"{np.std(returns_array) * np.sqrt(252) * 100:.2f}%"
        }
        
        return metrics
    
    def calculate_win_rate(self):
        """Calculate win rate of trades"""
        if len(self.trades) < 2:
            return 0
        
        wins = 0
        total = 0
        
        buy_price = None
        for trade in self.trades:
            if trade['action'] == 'BUY':
                buy_price = trade['price']
            elif trade['action'] == 'SELL' and buy_price:
                if trade['price'] > buy_price:
                    wins += 1
                total += 1
                buy_price = None
        
        return wins / total if total > 0 else 0

# Example usage:
# backtester = Backtester(initial_balance=10000)
# # ... execute trades during backtesting ...
# report = backtester.generate_report(final_price=150.0)
# print(report)`
    },
    step7: {
      title: 'Step 7: Complete Trading System Integration',
      description: 'Integrate all components into a complete trading system',
      code: `# Step 7: Complete Trading System Integration
import torch
import numpy as np
import pandas as pd
from datetime import datetime

class CompleteTradingSystem:
    """
    Complete AI-powered trading system integrating:
    1. Data collection and preprocessing
    2. Technical indicators
    3. LSTM price prediction
    4. RL trading strategy
    5. Risk management
    6. Backtesting
    """
    def __init__(self, symbol='AAPL', initial_balance=10000):
        self.symbol = symbol
        self.initial_balance = initial_balance
        
        # Components
        self.price_model = None
        self.rl_agent = None
        self.risk_manager = RiskManager()
        self.backtester = Backtester(initial_balance)
        
        # Data
        self.raw_data = None
        self.processed_data = None
        self.feature_columns = None
    
    def load_and_prepare_data(self, period='2y'):
        """Step 1-3: Load data and prepare features"""
        print("Step 1: Loading market data...")
        self.raw_data = fetch_stock_data(self.symbol, period)
        
        print("Step 2: Cleaning and preprocessing...")
        cleaned_data = clean_data(self.raw_data.copy())
        featured_data = add_basic_features(cleaned_data)
        
        print("Step 3: Adding technical indicators...")
        self.processed_data = add_technical_indicators(featured_data)
        self.processed_data, scaler = normalize_features(
            self.processed_data, 
            self.processed_data.columns.tolist()
        )
        
        self.feature_columns, _ = prepare_features(self.processed_data)
        print(f"Data prepared: {len(self.processed_data)} days, {len(self.feature_columns)} features")
        
        return self.processed_data
    
    def train_price_prediction_model(self, sequence_length=60):
        """Step 4: Train LSTM for price prediction"""
        print("Step 4: Training price prediction model...")
        
        train_data, val_data, test_data, _ = prepare_training_data(
            self.processed_data, 
            sequence_length=sequence_length
        )
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        self.price_model, _, _ = train_price_prediction_model(
            X_train, y_train, X_val, y_val,
            input_size=len(self.feature_columns),
            hidden_size=64,
            num_layers=2
        )
        
        print("Price prediction model trained!")
        return self.price_model
    
    def train_rl_agent(self, episodes=100):
        """Step 5: Train RL agent for trading strategy"""
        print("Step 5: Training RL trading agent...")
        
        # Prepare data for RL environment
        env_data = self.processed_data[self.feature_columns].values
        
        env = TradingEnvironment(env_data, self.initial_balance)
        self.rl_agent = DQNAgent(
            state_size=len(self.feature_columns) + 3,
            action_size=3
        )
        
        rewards = self.rl_agent.train(env, episodes=episodes)
        print("RL agent trained!")
        
        return rewards
    
    def predict_price(self, recent_data, sequence_length=60):
        """Predict next price using LSTM model"""
        if self.price_model is None:
            return None
        
        # Prepare input sequence
        if len(recent_data) < sequence_length:
            return None
        
        sequence = recent_data[-sequence_length:].values
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        self.price_model.eval()
        with torch.no_grad():
            prediction = self.price_model(sequence_tensor)
        
        return prediction.item()
    
    def get_trading_signal(self, current_state):
        """Get trading action from RL agent"""
        if self.rl_agent is None:
            return 0  # Hold
        
        action = self.rl_agent.act(current_state, training=False)
        return action
    
    def backtest_strategy(self, start_date=None, end_date=None):
        """Step 6: Backtest complete strategy"""
        print("Step 6: Backtesting strategy...")
        
        self.backtester.reset()
        
        # Filter data by date range if provided
        test_data = self.processed_data.copy()
        if start_date:
            test_data = test_data[test_data.index >= start_date]
        if end_date:
            test_data = test_data[test_data.index <= end_date]
        
        env_data = test_data[self.feature_columns].values
        prices = test_data['Close'].values
        
        # Simulate trading
        for i in range(len(env_data) - 1):
            current_state = self.get_state_from_data(env_data[i], i)
            action = self.get_trading_signal(current_state)
            
            current_price = prices[i]
            self.backtester.execute_trade(action, current_price, test_data.index[i])
            self.backtester.update_portfolio_value(current_price)
        
        # Final report
        final_price = prices[-1]
        report = self.backtester.generate_report(final_price)
        
        return report
    
    def get_state_from_data(self, market_features, step):
        """Create state vector from market data"""
        # Get portfolio state from backtester
        portfolio_value = self.backtester.portfolio_values[-1] if self.backtester.portfolio_values else self.initial_balance
        balance = self.backtester.balance
        shares = self.backtester.shares
        
        portfolio_state = np.array([
            balance / self.initial_balance,
            shares / 100,
            portfolio_value / self.initial_balance
        ])
        
        return np.concatenate([market_features, portfolio_state])
    
    def live_trading_simulation(self, new_data_point):
        """
        Simulate live trading with new data point
        
        This would be called in real-time with new market data
        """
        # Update processed data
        self.processed_data = pd.concat([self.processed_data, new_data_point])
        
        # Get recent sequence for prediction
        recent_features = self.processed_data[self.feature_columns].tail(60)
        
        # Predict price
        predicted_price = self.predict_price(recent_features)
        
        # Get current state
        current_state = self.get_state_from_data(
            recent_features.iloc[-1].values,
            len(self.processed_data) - 1
        )
        
        # Get trading signal
        action = self.get_trading_signal(current_state)
        
        # Apply risk management
        current_price = self.processed_data['Close'].iloc[-1]
        
        if action == 1:  # Buy
            stop_loss = self.risk_manager.apply_stop_loss(
                current_price, current_price, stop_loss_pct=0.05
            )
            position_size = self.risk_manager.calculate_position_size(
                self.backtester.portfolio_values[-1],
                current_price,
                stop_loss
            )
            # Execute trade with position sizing...
        
        return {
            'action': ['Hold', 'Buy', 'Sell'][action],
            'predicted_price': predicted_price,
            'current_price': current_price,
            'confidence': abs(predicted_price - current_price) / current_price if predicted_price else 0
        }

# 7.1 Complete Usage Example
def run_complete_trading_system():
    """Run complete trading system pipeline"""
    
    # Initialize system
    system = CompleteTradingSystem(symbol='AAPL', initial_balance=10000)
    
    # Step 1-3: Prepare data
    system.load_and_prepare_data(period='2y')
    
    # Step 4: Train price prediction model
    system.train_price_prediction_model(sequence_length=60)
    
    # Step 5: Train RL agent
    system.train_rl_agent(episodes=100)
    
    # Step 6: Backtest strategy
    report = system.backtest_strategy()
    
    print("\\n=== TRADING SYSTEM PERFORMANCE REPORT ===")
    for key, value in report.items():
        print(f"{key}: {value}")
    
    return system, report

# Run the complete system
# system, report = run_complete_trading_system()`
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-green-50 rounded-lg p-4 border-2 border-green-200 mb-4">
        <h2 className="text-2xl font-bold text-green-900 mb-2">AI Trading Tools - Complete Tutorial</h2>
        <p className="text-green-800">
          A comprehensive step-by-step guide to building AI-powered trading tools. 
          Learn data collection, technical indicators, LSTM price prediction, reinforcement learning strategies, 
          risk management, and complete system integration.
        </p>
      </div>

      {/* Framework Selector */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Select Framework
        </label>
        <select
          value={selectedFramework}
          onChange={(e) => setSelectedFramework(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
        >
          <option value="pytorch">PyTorch</option>
          <option value="tensorflow">TensorFlow</option>
        </select>
      </div>

      {/* Step Selector */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Select Tutorial Step
        </label>
        <select
          value={selectedStep}
          onChange={(e) => setSelectedStep(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
        >
          <option value="step1">Step 1: Data Collection & Preprocessing</option>
          <option value="step2">Step 2: Mathematical Foundations - Technical Indicators</option>
          <option value="step3">Step 3: Feature Engineering & Data Preparation</option>
          <option value="step4">Step 4: LSTM Model for Price Prediction</option>
          <option value="step5">Step 5: Reinforcement Learning Trading Strategy</option>
          <option value="step6">Step 6: Risk Management & Performance Metrics</option>
          <option value="step7">Step 7: Complete Trading System Integration</option>
        </select>
      </div>

      {/* Step Content */}
      <div className="bg-white rounded-lg p-6 shadow-md">
        <h3 className="text-xl font-bold text-gray-900 mb-2">
          {steps[selectedStep].title}
        </h3>
        <p className="text-gray-600 mb-4">
          {steps[selectedStep].description}
        </p>
        
        <div className="mt-4">
          <SyntaxHighlighter language="python" style={vscDarkPlus} showLineNumbers>
            {steps[selectedStep].code}
          </SyntaxHighlighter>
        </div>
      </div>
    </div>
  );
}

