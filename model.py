import torch
import torch.nn as nn

class Head(nn.Module):

    def __init__(self, n_embd, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
    

class GPT(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

    def forward(self, x):
        token_embeddings = self.token_embedding(x)

        position_embeddings = self.position_embedding(
            torch.arange(x.size(1), device=x.device)
        )

        x = token_embeddings + position_embeddings

        return x

model = GPT(
    vocab_size=512,
    block_size=128,
    n_embd=128
)

x = torch.tensor([[10, 20, 30, 40]])

output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)