"""Evaluate the full rank and low rank models on MNIST dataset.
"""
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import the data distribured parallel module from torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from models import FullRankNet, LowRankNet, convert_to_low_rank

import logging

# Set the seed
torch.manual_seed(0)
# Evaluate whether cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


logger.info(f'Using device {device}')


def evaluate_accuracy(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def test_loss(net, test_loader):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # Transform the labels to one hot encoding
            labels = torch.nn.functional.one_hot(
                labels, 10).float()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total += 1
    return total_loss / total


if __name__ == '__main__':
    # Download the mnist dataset
    logger.info('Downloading the MNIST dataset')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create the full rank network
    full_rank_net = FullRankNet(no_hidden_layers=4, hidden_size=128).to(device)

    # Print the accuract of the model and the cross entropy loss
    full_rank_net.load_state_dict(torch.load('models/full_rank_net.pth'))

    accuracy = evaluate_accuracy(full_rank_net, test_loader)
    logger.info(f'Accuracy of the full rank model is {accuracy}')

    loss = test_loss(full_rank_net, test_loader)
    logger.info(f'Cross entropy loss of the full rank model is {loss}')
