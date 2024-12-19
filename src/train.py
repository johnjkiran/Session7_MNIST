import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import Model_1
from tqdm import tqdm
import datetime
import os

try:
    from torchsummary import summary
except ImportError:
    from torchinfo import summary

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)
        y = torch.randint(h, (1,))
        x = torch.randint(w, (1,))

        y1 = torch.clamp(y - self.length // 2, 0, h)
        y2 = torch.clamp(y + self.length // 2, 0, h)
        x1 = torch.clamp(x - self.length // 2, 0, w)
        x2 = torch.clamp(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.
        mask = mask.expand_as(img)
        img = img * mask
        return img

def data_set(cuda):    
    # Train Phase transformations
    train_transforms = transforms.Compose([
                                        #  transforms.Resize((28, 28)),
                                        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                        # transforms.RandomAffine(
                                        #     degrees=0, 
                                        #     translate=(0.08, 0.08),
                                        #     scale=(0.98, 1.02),
                                        #     shear=(-1, 1)
                                        # ),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                        Cutout(length=4)
                                        ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
                                        #  transforms.Resize((28, 28)),
                                        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])
    
    train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    return train_loader, test_loader

def model_param():
    SEED = 1
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model
    print(device)
    model = Model_1().to(device)
    
    # Print model summary
    summary(model, input_size=(1, 28, 28))
    return cuda, model, device

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))
    return 100. * correct / len(test_loader.dataset)

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if __name__ == "__main__":
    cuda, model, device = model_param()
    train_loader, test_loader = data_set(cuda)
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    # scheduler = StepLR(optimizer, step_size=8, gamma=0.5)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.4)

    EPOCHS = 15
    timestamp = get_timestamp()
    model_save_path = f"models/mnist_model_{timestamp}.pth"
    best_accuracy = 0.0
    
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        accuracy = test(model, device, test_loader)
        
        # Save model if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'timestamp': timestamp
            }, model_save_path)
            print(f"Model saved to {model_save_path}")
