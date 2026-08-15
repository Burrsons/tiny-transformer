import torch

embeddings = torch.tensor([
    [0.2, -0.7, 0.4, 0.9],   # cat
    [0.3, -0.6, 0.5, 0.8],   # dog
    [-0.1, 0.8, 0.2, -0.4]    # car
])

print(embeddings)
print("Shape:", embeddings.shape)