import torch

# Word embeddings 
embeddings = torch.tensor([
    [0.2, -0.7, 0.4, 0.9],    # cat
    [0.3, -0.6, 0.5, 0.8],    # dog
    [-0.1, 0.8, 0.2, -0.4]    # car
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

# Create weight matrices for Query, Key and Value
W_q = torch.tensor([
    [0.5, 0.1],
    [0.2, 0.4],
    [0.3, 0.2],
    [0.1, 0.6],
])

W_k = torch.tensor([
    [0.4, 0.2], 
    [0.1, 0.5], 
    [0.3, 0.1], 
    [0.1, 0.6]  
])

W_v = torch.tensor([
    [0.6, 0.1], 
    [0.2, 0.5], 
    [0.4, 0.2], 
    [0.1, 0.3]  
])

# Create queries, keys and values using matrix multiplication
queries = embeddings @ W_q
keys = embeddings @ W_k
values =  embeddings @ W_v

print("\nQueries:")
print(queries)
print("Shape:", queries.shape)

print("\nKeys:") 
print(keys)
print("Shape:", keys.shape)

print("\nValues:") 
print(values)
print("Shape:", values.shape)

# Calculate scaled attention scores
d_keys = keys.shape[1]

scores = (queries @ keys.t()) / torch.sqrt(
    torch.tensor(d_keys, dtype=torch.float32)
)

print("\n Scaled attention scores:")
print(scores)

# Convert scores into attention weights using Softmax
attention_weights = torch.softmax(scores, dim=1)

print("\nAttention weights:")
print(attention_weights)

# Create context vectors
context = attention_weights @ values

print("\nContext vectors:")
print(context)