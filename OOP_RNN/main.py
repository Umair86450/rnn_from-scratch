# main.py
from simple_rnn import SimpleRNN

# Training data
data = [
    "hello how are you",
    "how are you doing",
    "are you doing well"
]

# Vocabulary
words = sorted(set(" ".join(data).split()))
word_to_idx = {w: i for i, w in enumerate(words)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

# Initialize and train
rnn = SimpleRNN(vocab_size=len(words))
rnn.train(data, word_to_idx)

# Predict
print("\n--- Predictions ---\n")
for sentence in data:
    rnn.predict(sentence, word_to_idx, idx_to_word)
