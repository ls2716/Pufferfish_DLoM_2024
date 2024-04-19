"""Train the full rank and low rank models on MNIST dataset.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import the data distribured parallel module from torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR

from models import FullRankNet, LowRankNet, convert_to_low_rank
from evaluate_full import evaluate_accuracy, test_loss

from torch.utils.data.distributed import DistributedSampler

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

# EPOCHS = 30
# NO_HIDDEN_LAYERS = 4
# HIDDEN_SIZE = 128
# LOW_RANK = 16
# FIRST_LAYER_TO_LOW = 0

# Big model
EPOCHS = 3
NO_HIDDEN_LAYERS = 8
HIDDEN_SIZE = 512
LOW_RANK = 64
FIRST_LAYER_TO_LOW = 0


def compute_accuracy(net, val_loader, rank):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(rank), labels.to(rank)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train(net, train_loader, val_loader, epochs, rank, filename=None):
    times = []
    # Train the full rank network
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    last_accuracy = 0
    for epoch in range(epochs):
        net.train()
        train_loader.sampler.set_epoch(epoch)  # Set the epoch for the sampler
        # Log size of the train loader
        if rank == 0:
            logger.info(f"Rank {rank}: train_loader size {len(train_loader)}")
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                logger.info(f'Epoch {epoch}, Loss {loss.item()}')
        scheduler.step()
        # Log the training rate
        if rank == 0:
            logger.info(
                f"Rank {rank}: Learning rate {scheduler.get_last_lr()}")
            logger.info(
                f"Rank {rank}: Time taken for epoch {epoch} is {time.time() - start_time}")

        times.append(time.time() - start_time)
        # Log the current accuracy
        accuracy = compute_accuracy(net, val_loader, rank)
        if rank == 0:
            logger.info(f"Rank {rank}: Validation accuracy {accuracy}")
            if accuracy > last_accuracy:
                logger.info('Saving the model')
                torch.save(net.module.state_dict(), 'models/'+filename)
                last_accuracy = accuracy
    return times


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare(rank, world_size, train_dataset, batch_size=64, pin_memory=False, num_workers=0):

    sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=pin_memory,
                              num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)

    return train_loader


def train_ddp(rank, world_size):

    setup(rank, world_size)

    logger.info(f"Rank {rank} started")
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
    # split the test dataset into test and validation
    test_dataset, val_dataset = torch.utils.data.random_split(
        test_dataset, [int(0.5*len(test_dataset)), int(0.5*len(test_dataset))])

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    train_loader = prepare(rank, world_size, train_dataset)

    full_rank_net = FullRankNet(no_hidden_layers=NO_HIDDEN_LAYERS,
                                hidden_size=HIDDEN_SIZE)
    # Create low rank model
    model = LowRankNet(full_rank_net=full_rank_net,
                       low_rank=LOW_RANK,
                       first_layer_to_low=FIRST_LAYER_TO_LOW).to(rank)
    if rank == 0:
        # Print model summary
        logger.info(model)
        # Compute the number of model parameters
        logger.info(
            f"Rank {rank}: Number of model parameters {sum(p.numel() for p in model.parameters())}")
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer

    times = train(ddp_model, train_loader, val_loader, epochs=EPOCHS,
                  rank=rank, filename='low_rank_net.pth')

    # Log the mean time per epoch
    logger.info(f"Rank {rank}: Mean time per epoch {sum(times)/len(times)}")

    cleanup()


if __name__ == '__main__':

    world_size = 4
    start_time = time.time()
    mp.spawn(train_ddp,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    print(f"Time taken for training is {time.time() - start_time}")

    # Load the low rank model and evaluate the accuracy
    low_rank_net = LowRankNet(full_rank_net=FullRankNet(
        no_hidden_layers=NO_HIDDEN_LAYERS, hidden_size=HIDDEN_SIZE),
        low_rank=LOW_RANK, first_layer_to_low=FIRST_LAYER_TO_LOW).to(device)
    low_rank_net.load_state_dict(torch.load('models/low_rank_net.pth'))

    # Download the mnist  test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    accuracy = evaluate_accuracy(low_rank_net, test_loader)
    logger.info(f'Accuracy of the low rank model is {accuracy}')

    loss = test_loss(low_rank_net, test_loader)
    logger.info(f'Cross entropy loss of the low rank model is {loss}')
