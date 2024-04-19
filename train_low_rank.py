"""Train the full rank and low rank models on MNIST dataset.
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# Import plateou scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import the data distribured parallel module from torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from models import FullRankNet, LowRankNet, convert_to_low_rank
from evaluate_full import evaluate_accuracy, test_loss

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


def train(net, train_loader, val_loader, epochs):
    def validate(net, val_loader):
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total
    # Train the full rank network
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.1, patience=2, mode='max')
    last_val_acc = 0
    last_improvement = 0
    for epoch in range(epochs):
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                logger.info(f'Epoch {epoch}, Loss {loss.item()}')
        val_acc = validate(net, val_loader)
        logger.info(f'Validation accuracy {val_acc}')
        scheduler.step(val_acc)  # Update the learning rate
        logger.info(f'Learning rate {scheduler.get_last_lr()}')

        # Save the full rank model
        if val_acc > last_val_acc:
            logger.info('Saving the model')
            torch.save(net.state_dict(), 'models/low_rank_net.pth')
            last_improvement = epoch
            last_val_acc = val_acc
        if epoch - last_improvement > 5:
            break


if __name__ == '__main__':
    # Get the seed from the command line
    seed = int(sys.argv[1])
    torch.manual_seed(seed)
    logger.info(f'Seed set to {seed}')
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

    # Split the test dataset into two parts
    test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [
        5000, 5000])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    total_epochs = 40
    warmup_epochs = 2

    # Create the full rank network
    full_rank_net = FullRankNet(no_hidden_layers=4, hidden_size=128).to(device)

    # Train the full rank network
    train(full_rank_net, train_loader, val_loader, warmup_epochs)

    # Save the full rank model
    torch.save(full_rank_net.state_dict(), 'models/full_rank_net_warmup.pth')

    # Initialise the low rank net
    low_rank_net = LowRankNet(
        full_rank_net, low_rank=32, first_layer_to_low=1).to(device)

    # Convert the full rank network to a low rank network
    convert_to_low_rank(full_rank_net, low_rank_net)
    logger.info('Converted the full rank network to a low rank network')

    # Train the low rank network
    train(low_rank_net, train_loader, val_loader, total_epochs - warmup_epochs)

    # Print the accuracy of the model and the cross entropy loss
    low_rank_net.load_state_dict(torch.load('models/low_rank_net.pth'))

    accuracy = evaluate_accuracy(low_rank_net, test_loader)
    logger.info(f'Accuracy of the low rank model is {accuracy}')

    loss = test_loss(low_rank_net, test_loader)
    logger.info(f'Cross entropy loss of the low rank model is {loss}')
