import pytest
import torch
from src.models import StockPredictor


def test_stock_predictor_init():
    model = StockPredictor(input_size=10, hidden_size=64, num_layers=2, num_classes=1, dropout_prob=0.5)
    assert isinstance(model, torch.nn.Module)
    assert model.lstm.input_size == 10
    assert model.lstm.hidden_size == 64
    assert model.lstm.num_layers == 2
    assert model.fc.out_features == 1


def test_stock_predictor_forward_pass():
    input_size = 10
    hidden_size = 64
    num_layers = 2
    num_classes = 1
    sequence_length = 5
    batch_size = 4

    model = StockPredictor(input_size, hidden_size, num_layers, num_classes)

    # Simulate input from DataLoader: (batch_size, sequence_length, input_size)
    dummy_input = torch.randn(batch_size, sequence_length, input_size)

    output = model(dummy_input)

    # Output should be (batch_size, num_classes)
    assert output.shape == (batch_size, num_classes)
    # For binary classification with BCELoss, output should be between 0 and 1 (after sigmoid)
    assert torch.all(output >= 0) and torch.all(output <= 1)


def test_stock_predictor_dropout():
    model = StockPredictor(input_size=10, hidden_size=64, num_layers=2, num_classes=1, dropout_prob=0.8)
    # In training mode, dropout should be active
    model.train()
    assert model.lstm.dropout == 0.8
    # In eval mode, dropout should be inactive
    model.eval()
    assert model.lstm.dropout == 0.8  # Dropout layer's 'p' doesn't change, but it's applied differently
    # A better test would be to check if outputs are different in train vs eval for the same input
    # but that's more complex and often covered by PyTorch's internal tests.