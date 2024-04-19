# Testing Pufferfish on the task of training FCN on MNIST dataset

Author: Lukasz Sliwinski s1640204@ed.ac.uk

As a part of the assessment for the Deep Learing on Manifolds module (Spring 2024).

### What has been done

In this directory, I have implemented the [Pufferfish](https://proceedings.mlsys.org/paper_files/paper/2021/hash/94cb28874a503f34b3c4a41bddcea2bd-Abstract.html) algorithm on the task of training a fully-connected network (FCN) on the MNIST dataset. Description of the files:

- models.py - contains implementation of full rank FCN, low rank FCN together with implementation of a low rank layer, and the implementation of conversion from the full rank network to the low rank network. Below is a code that converts a single fully-connected linear layer to a low rank layer:

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

- train_full_rank.py contains the training of the full rank network
- train_low_rank.py implements the Pufferfish training with hybrid networks and vanilla warm-up
  
Because of the high number of hyperparameters, I have not done any experiments with the above files - they just show that I have done the work to implement the algorithm.

- train_full_DDP.py implements DataDistributedParallel training of a full rank network
- train_low_DDP.py implements DataDistributedParallel training of a low rank network

Using above scripts, I have tested whether Pufferfish indeed leads to time savings and what is the order of the loss due to direct training of a low rank network. A few results from experimentation with those scripts will be described in the following sections.

- data folder is for data (empty in this repo)
- models folder holds models (empty in this repo)
- logs folder contains logs from running the aforementioned scripts. The output from log files will be used to describe the results.

Other files are auxilary.

## Data-Distributed Parallel (DDP) training

The DDP training was executed on MAC-MIGS GPU cluster which provides 4 x T600 NVIDIA GPUs. All runs used all 4 gpus.

## Time savings and accuracy for a small model

The small full rank model was:
```bash
2024-04-19 14:51:43,888 - __mp_main__ - INFO - FullRankNet(
  (input_layer): Linear(in_features=784, out_features=128, bias=True)
  (hidden_layers): ModuleList(
    (0-3): 4 x Linear(in_features=128, out_features=128, bias=True)
  )
  (output_layer): Linear(in_features=128, out_features=10, bias=True)
)
```

That is apart from input dimension with 28x28 and output dimension 10, hidden_dimension was 128. There were 4 hidden fully-connected linear layers.

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
Only the hidden linear layers were decomposed into low rank layers and the low rank dimenion was 16.

#### Number of parameters

The full rank model had 167818 parameters while the low rank model had 118666 parameters which is around 30\% less.

#### Training

The models were trained for 30 epochs using Adam optimiser and learning rate 0.001 reduced 10 times every 10 epochs. The best validation accuracy model was used to compute the final accuracy on a test set.

#### Final accuracy and mean epoch time

Test accuracy:

- full rank model: 0.978
- low rank model: 0.9727
  
Mean time per epoch:

- full rank model: ~6.4s
- low rank model: ~6.4s (no non-negligible time saving)

The mean times were computed across three runs.

We can see that mean time per epoch was almost the same for both models.

## Time savings and accuracy for a large model

The large full rank model was:
```bash
2024-04-19 15:15:41,214 - __mp_main__ - INFO - FullRankNet(
  (input_layer): Linear(in_features=784, out_features=512, bias=True)
  (hidden_layers): ModuleList(
    (0-7): 8 x Linear(in_features=512, out_features=512, bias=True)
  )
  (output_layer): Linear(in_features=512, out_features=10, bias=True)
)
```

That is apart from input dimension with 28x28 and output dimension 10, hidden_dimension was 512. There were 8 hidden layers.

The corresponding low rank model was:
```bash
2024-04-19 15:18:10,928 - __mp_main__ - INFO - LowRankNet(
  (input_layer): Linear(in_features=784, out_features=512, bias=True)
  (hidden_layers): ModuleList(
    (0-7): 8 x LowRankLayer(
      (u_layer): Linear(in_features=512, out_features=64, bias=False)
      (v_layer): Linear(in_features=64, out_features=512, bias=True)
    )
  )
  (output_layer): Linear(in_features=512, out_features=10, bias=True)
)
```
Only the inner linear layers were decomposed into low rank layers and the low rank dimenion was 64.

#### Number of parameters

The full rank model had 2508298 parameters while the low rank model had 935434 parameters which is around 63\% less.

#### Training

The models were trained for 3 epochs using Adam optimiser.

Naturally, the output models were not fully trained. The purpose of the experiments was to calculate and compare the mean time per epoch.

#### Mean epoch time
  
Mean time per epoch:

- full rank model: ~12.2s
- low rank model: ~8.5s (3.7s saving - around 30\% less)

The mean times were computed across three runs.

## Discussion

The main conclusion based on the results is that Pufferfish does not always provide time savings.

Because low rank decomposition of a linear layer extends the computational graph of the network, less parameters of the network does not necesarily imply less computation.

In the case of the small model, although the low rank network had 30\% less parameters than the full rank network, the additional computation time due to higher number of layers reduced any time savings from having a decreased number of parameters (that is, savings from reduced communication overhead and less gradients to be computed).

In the case of the large model, the low rank network has over 60\% less parameters than the full rank network. Consequently, this results in reduction of mean time per epoch by around 30\%.

We can thus conclude that low-rank decomposition of linear layers does not always lead to a decrease in the training time of the network. Only when the relative difference in the number of parameters between the full rank and the low rank models is large, it is possible to obtain positive reduction in computation time.

#### Other considerations

As mentioned in the executive summary, in addition to the problem above, we are faced with hyperparameter optimisation for the number of warm-up epochs and the number of layers to be decomposed.

Furthermore, the low rank representation does poorly in terms of decreasing the total number of operations which is an important performance metric as it determines the inference efficiency of the model.

All of above makes the application of direct low rank training, and thus the Pufferfish algorithm, a complicated task.

##### Addendum

While I wrote the code myself, I naturally read through the code repository correponding to the paper available at [https://github.com/hwang595/Pufferfish](https://github.com/hwang595/Pufferfish).
