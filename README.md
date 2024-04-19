# Testing Pufferfish for the task of training MLP on MNIST dataset

Author: Lukasz Sliwinski s1640204@ed.ac.uk

As a part of assessment for Deep Learing on Manifolds module Spring 2024

### What has been done

In this directory, I have implemented the Pufferfish algorithm on the task of training MLP on the MNIST dataset. Description of the files:

- models.py - contains implementation of full rank mlp, low rank mlp together with implementation of a low rank layer, and the implementation of conversion from full rank network to low rank network. Below is a code that converts a single linear layer to a low rank layer

```python
# Get the matrix
A = full_rank_net.hidden_layers[i].weight.data
# Perform SVD
U, S, Vh = torch.linalg.svd(A)
# Compute new U
U_low_rank = U[:, :low_rank] @ torch.sqrt(torch.diag(S[:low_rank]))
# Compute new Vh
Vh_low_rank = torch.sqrt(torch.diag(
    S[:low_rank])) @ Vh[:low_rank, :]
# Copy the weights of the low rank layer
low_rank_net.hidden_layers[i].u_layer.weight.data[:] =\
    U_low_rank.T[:]
low_rank_net.hidden_layers[i].v_layer.weight.data[:] = \
    Vh_low_rank.T[:]
# Copy the biases of the low rank layer
low_rank_net.hidden_layers[i].v_layer.bias.data[:] = \
    full_rank_net.hidden_layers[i].bias.data[:]
```

- train_full_rank.py contains the training of vanilla network
- train_low_rank.py implements the Pufferfish training with hybrid networks and vanilla warm-up
  
Because of the high number of hyperparameters, I have not done any experiments with the above files - they just show that I have done the work to implement main part of the algorithm.

- train_full_DDP.py implements DataDistributedParallel training of a full rank network
- train_low_DDP.py implements DataDistributedParallel training of a low rank network

Using above scripts, I have analysed whether Pufferfish indeed leads to time savings and what is the order of the loss due to direct training of a low rank network. A few results from experimentation with those scripts will be described in the following sections.

- data folder is for data (empty in this repo)
- models folder holds models (empty in this repo)
- logs folder contains logs from running the aforementioned scripts. The output from log files will be used to describe the results.

Other files are auxilary.

## DDP training

The DDP training was executed on MAC-MIGS GPU cluster which provides 4 x T600 NVIDIA GPUs. All runs used all 4 gpus.

## Time savings and accuracy for small model

The small full rank model is:
```bash
2024-04-19 14:51:43,888 - __mp_main__ - INFO - FullRankNet(
  (input_layer): Linear(in_features=784, out_features=128, bias=True)
  (hidden_layers): ModuleList(
    (0-3): 4 x Linear(in_features=128, out_features=128, bias=True)
  )
  (output_layer): Linear(in_features=128, out_features=10, bias=True)
)
```

That is apart from input dimension with 28x28 and output dimension 10, hidden_dimension was 128.

The corresponding low rank model was:
```bash
2024-04-19 14:33:47,435 - __mp_main__ - INFO - LowRankNet(
  (input_layer): Linear(in_features=784, out_features=128, bias=True)
  (hidden_layers): ModuleList(
    (0-3): 4 x LowRankLayer(
      (u_layer): Linear(in_features=128, out_features=16, bias=False)
      (v_layer): Linear(in_features=16, out_features=128, bias=True)
    )
  )
  (output_layer): Linear(in_features=128, out_features=10, bias=True)
)
```
Only the inner linear layers were rendered low rank and the low rank dimenion was 16.

#### Number of parameters

The full rank model had 167818 parameters while the low rank model had 118666 parameters which is around 30\% less.

#### Training

The models were trained for 30 epochs using Adam optimiser and learning rate 0.001 reduced 10 times every 10 epochs. The best validation accuracy model was used to compute the final accuracy on a test set.

#### Final accuracy and mean epoch time

Accuracy:

- full rank model: 0.978
- low rank model: 0.9727
  
Mean time per epoch:

- full rank model: 
- low rank model:

The mean times were computed across three runs.

## Time savings and accuracy for large model

The small full rank model is:
```bash
2024-04-19 14:51:43,888 - __mp_main__ - INFO - FullRankNet(
  (input_layer): Linear(in_features=784, out_features=128, bias=True)
  (hidden_layers): ModuleList(
    (0-3): 4 x Linear(in_features=128, out_features=128, bias=True)
  )
  (output_layer): Linear(in_features=128, out_features=10, bias=True)
)
```

That is apart from input dimension with 28x28 and output dimension 10, hidden_dimension was 128.

The corresponding low rank model was:
```bash
2024-04-19 14:33:47,435 - __mp_main__ - INFO - LowRankNet(
  (input_layer): Linear(in_features=784, out_features=128, bias=True)
  (hidden_layers): ModuleList(
    (0-3): 4 x LowRankLayer(
      (u_layer): Linear(in_features=128, out_features=16, bias=False)
      (v_layer): Linear(in_features=16, out_features=128, bias=True)
    )
  )
  (output_layer): Linear(in_features=128, out_features=10, bias=True)
)
```
Only the inner linear layers were rendered low rank and the low rank dimenion was 16.

#### Number of parameters

The full rank model had 167818 parameters while the low rank model had 118666 parameters which is around 30\% less.

#### Training

The models were trained for 30 epochs using Adam optimiser and learning rate 0.001 reduced 10 times every 10 epochs. The best validation accuracy model was used to compute the final accuracy on a test set.

#### Final accuracy and mean epoch time

Accuracy:

- full rank model: 0.978
- low rank model: 0.9727
  
Mean time per epoch:

- full rank model: 
- low rank model:

The mean times were computed across three runs.