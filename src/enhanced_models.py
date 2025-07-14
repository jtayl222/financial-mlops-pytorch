"""
Enhanced model architectures to improve performance from 52.7% to 78%+
Focus on better architectures for financial time series prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EnhancedStockPredictor(nn.Module):
    """Enhanced LSTM with attention mechanism and better architecture"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes=1, dropout_prob=0.3):
        super(EnhancedStockPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM for better context
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_prob, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size * 2)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Feature extraction
        features = self.feature_extractor(attended_output)
        
        # Classification
        output = self.classifier(features)
        
        return output

class TransformerStockPredictor(nn.Module):
    """Transformer-based model for better financial time series prediction"""
    
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=6, num_classes=1, dropout_prob=0.1):
        super(TransformerStockPredictor, self).__init__()
        self.d_model = d_model
        self.input_size = input_size
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout_prob)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_prob,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(d_model // 2, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer expects (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Transformer forward pass
        transformer_out = self.transformer(x)  # (seq_len, batch_size, d_model)
        
        # Use last time step for classification
        output = transformer_out[-1]  # (batch_size, d_model)
        
        # Classification
        output = self.classifier(output)
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EnsembleStockPredictor(nn.Module):
    """Ensemble of multiple models for better performance"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes=1, dropout_prob=0.3):
        super(EnsembleStockPredictor, self).__init__()
        
        # Multiple base models
        self.lstm_model = EnhancedStockPredictor(input_size, hidden_size, num_layers, 
                                               num_classes, dropout_prob)
        self.transformer_model = TransformerStockPredictor(input_size, d_model=hidden_size, 
                                                         num_classes=num_classes, 
                                                         dropout_prob=dropout_prob)
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.tensor([0.6, 0.4]))
        
    def forward(self, x):
        # Get predictions from both models
        lstm_out = self.lstm_model(x)
        transformer_out = self.transformer_model(x)
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_out = weights[0] * lstm_out + weights[1] * transformer_out
        
        return ensemble_out

class CNNLSTMStockPredictor(nn.Module):
    """CNN-LSTM hybrid for capturing both local patterns and temporal dependencies"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes=1, dropout_prob=0.3):
        super(CNNLSTMStockPredictor, self).__init__()
        
        # CNN for local pattern extraction
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(64, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_prob)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.size()
        
        # CNN expects (batch_size, channels, seq_len)
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)
        
        # CNN forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Back to (batch_size, seq_len, features)
        x = x.transpose(1, 2)  # (batch_size, seq_len, 64)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use last time step
        output = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(output)
        
        return output

def get_enhanced_model(model_type, input_size, hidden_size, num_layers, num_classes=1, dropout_prob=0.3):
    """Factory function to get enhanced models"""
    
    if model_type == "enhanced_lstm":
        return EnhancedStockPredictor(input_size, hidden_size, num_layers, num_classes, dropout_prob)
    elif model_type == "transformer":
        return TransformerStockPredictor(input_size, d_model=hidden_size, num_classes=num_classes, dropout_prob=dropout_prob)
    elif model_type == "ensemble":
        return EnsembleStockPredictor(input_size, hidden_size, num_layers, num_classes, dropout_prob)
    elif model_type == "cnn_lstm":
        return CNNLSTMStockPredictor(input_size, hidden_size, num_layers, num_classes, dropout_prob)
    else:
        raise ValueError(f"Unknown model type: {model_type}")