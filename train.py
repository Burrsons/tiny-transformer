import torch

# Word embeddings
embeddings = torch.tensor([
    [0.2, -0.7, 0.4, 0.9],    # cat
    [0.3, -0.6, 0.5, 0.8],     # dog
    [-0.1, 0.8, 0.2, -0.4]     # car
])

# Display embeddings and shape
print("Embeddings:")
print(embeddings)
print("Shape:", embeddings.shape)

# Extract individual embeddings
cat = embeddings[0]
dog = embeddings[1]
car = embeddings[2]

print("\nIndividual embeddings:")
print("cat:", cat)
print("dog:", dog)
print("car:", car)

# Calculate pairwise dot products
print("\nDot products:")
print("cat · dog:", torch.dot(cat, dog))
print("cat · car:", torch.dot(cat, car))
print("dog · car:", torch.dot(dog, car))

# Calculate all attention scores at once
scores = embeddings @ embeddings.T

print("\nAttention scores:")
print(scores)

# Convert scores into attention weights
attention_weights = torch.softmax(scores, dim=1)

print("\nAttention weights:")
print(attention_weights)

# Create context vectors
context = attention_weights @ embeddings

print("\nContext vectors:")
print(context)