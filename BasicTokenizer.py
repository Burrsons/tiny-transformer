def get_stats(ids):
    counts = {}

    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])

        if pair in counts:
            counts[pair] += 1
        else:
            counts[pair] = 1
    return counts

def merge(ids, pair, new_id):
    new_ids = []
    i = 0

    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids
class BasicTokenizer:
    
    def train(self, text, vocab_size, verbose=True):
        assert vocab_size >= 256

        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges): 
            stats = get_stats(ids)

            if not stats:
                break

    
            pair = max(stats, key=stats.get)

            new_id = 256 + i

            ids = merge(ids, pair, new_id)

            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1}: {pair} -> {new_id}")
       
    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")
       

    def encode(self, text):
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            new_id = self.merges[pair]
            ids = merge(ids, pair, new_id)

        return ids

# Test code
with open("white_nights.txt", "r", encoding="utf-8") as f:
     text = f.read()

tokenizer = BasicTokenizer()

tokenizer.train(
    text,
    vocab_size=512,
    verbose=True
)
sample = "Nastenka"

encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)

print("Enocded:", encoded) 
print("Number of tokens:", len(encoded))
print("Decoded:", decoded)
print("Matches original:", decoded == sample)