import random
from BasicTokenizer import BasicTokenizer

def get_batch(tokens, block_size):
    start = random.randint(0, len(tokens) - block_size -1)

    x = tokens[start:start + block_size]
    y = tokens[start + 1:start + block_size +1]

    return x,y

with open("white_nights.txt", "r", encoding="utf-8") as f:
    text = f.read()


tokenizer = BasicTokenizer()

tokenizer.train(
    text,
    vocab_size=512,
    verbose=False
)

tokens = tokenizer.encode(text)

print("Number of tokens:", len(tokens))

x, y = get_batch(tokens, 128)

print("x:", x)
print("y:", y)