"""Test SVD of PyTorch tensors."""


import torch


# Create three random matrices for SVD
preU = torch.randn(5, 5)
preS = torch.diag(torch.randn(5))
preV = torch.randn(5, 5)


A = preU @ preS @ preV.t()

# Perform SVD
U, S, Vh = torch.linalg.svd(A)

# Reconstruct the matrix
A_reconstructed = U @ S.diag() @ Vh

print(A - A_reconstructed)
# Check if the reconstruction is correct
assert torch.allclose(A, A_reconstructed)


low_rank = 2
# Perform low rank approximation
U_low_rank = U[:, :low_rank]
S_low_rank = S[:low_rank]
Vh_low_rank = Vh[:low_rank, :]
print(U_low_rank.shape, S_low_rank.shape, Vh_low_rank.shape)

A_low_rank = U_low_rank @ S_low_rank.diag() @ Vh_low_rank

# Check if the low rank approximation is correct
print(A_low_rank - A)
assert torch.allclose(A, A_low_rank)
