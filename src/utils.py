import torch
import os

def load_latest_model(model_class):
    """Load the latest trained model based on timestamp"""
    model_files = [f for f in os.listdir('.') if f.startswith('model_') and f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError("No model files found")
    
    latest_model = max(model_files)
    model = model_class()
    model.load_state_dict(torch.load(latest_model))
    return model 