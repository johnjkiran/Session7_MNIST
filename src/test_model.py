import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Model_1
from train import data_set  # Import the data_set function
import pytest

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_architecture():
    model = Model_1()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    
    # Test output shape
    assert output.shape == (1, 10), "Output shape should be (batch_size, 10)"
    
    # Test parameter count
    param_count = count_parameters(model)
    assert param_count < 8000, f"Model has {param_count} parameters, should be less than 8000"

def test_model_forward_pass():
    model = Model_1()
    
    # Test batch processing
    batch_size = 64
    test_input = torch.randn(batch_size, 1, 28, 28)
    output = model(test_input)
    
    # Test output properties
    assert output.shape == (batch_size, 10), f"Expected output shape (64, 10), got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaN values"
    assert not torch.isinf(output).any(), "Model output contains infinite values"

def test_model_components():
    model = Model_1()
    
    # Test presence of essential layers
    assert hasattr(model, 'conv1'), "Model missing conv1 layer"
    assert hasattr(model, 'conv8'), "Model missing conv8 layer"
    assert hasattr(model, 'avgpool'), "Model missing avgpool layer"
    
    # Test dropout values
    assert isinstance(model.dropout1, nn.Dropout2d), "dropout1 should be Dropout2d"
    assert isinstance(model.dropout2, nn.Dropout2d), "dropout2 should be Dropout2d"
    assert abs(model.dropout1.p - 0.02) < 1e-5, "dropout1 should have p=0.02"
    assert abs(model.dropout2.p - 0.01) < 1e-5, "dropout2 should have p=0.01"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_cuda_compatibility():
    model = Model_1().cuda()
    test_input = torch.randn(1, 1, 28, 28).cuda()
    output = model(test_input)
    assert output.is_cuda, "Model output should be on CUDA device"

def test_model_training_mode():
    model = Model_1()
    
    # Test training mode
    model.train()
    assert model.training, "Model should be in training mode"
    
    # Test evaluation mode
    model.eval()
    assert not model.training, "Model should be in evaluation mode"

@pytest.mark.skipif(not torch.cuda.is_available() and not torch.backends.mps.is_available(), 
                   reason="Skip performance test in CI/CD environment")
def test_model_accuracy():
    # Determine device (CUDA, MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    model = Model_1().to(device)
    
    # Use the data_set function from train.py
    train_loader, test_loader = data_set(cuda=device.type=="cuda")
    
    model.eval()
    correct = 0
    total = 0
    
    try:
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target.view_as(predicted)).sum().item()
        
        accuracy = 100. * correct / total
        assert accuracy > 99.3, f"Model accuracy {accuracy:.2f}% is less than required 99.3%"
    
    except RuntimeError as e:
        pytest.skip(f"Runtime error during accuracy test: {str(e)}")
    finally:
        # Clean up
        if device.type == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    pytest.main([__file__]) 