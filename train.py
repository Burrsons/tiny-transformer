import torch

embeddings = torch.tensor([
    [0.2, -0.7, 0.4, 0.9],   # cat
    [0.3, -0.6, 0.5, 0.8],   # dog
    [-0.1, 0.8, 0.2, -0.4]    # car
])

print(embeddings)
print("Shape:", embeddings.shape)

cat = embeddings[0]
dog = embeddings[1]
car = embeddings[2]

score = torch.dot(cat,dog)

print("cat", cat)
print("dog", dog)
print("car", car)

print("cat · dog:", torch.dot(cat, dog))
print("cat · car:", torch.dot(cat, car))
print("dog · car:", torch.dot(dog, car))

print("cat · dog:", torch.dot(cat, dog))
print("cat · car:", torch.dot(cat, car))
print("dog · car:", torch.dot(dog, car))

scores = embeddings @ embeddings.T

print("Scores:")
print(scores)