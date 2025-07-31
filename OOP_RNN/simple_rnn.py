# simple_rnn.py
from base_rnn import RNNBase

class SimpleRNN(RNNBase):
    def __init__(self, vocab_size, hidden_size=8, learning_rate=0.1):
        # Call parent constructor
        super().__init__(vocab_size, hidden_size)
        self.learning_rate = learning_rate

    def train(self, data, word_to_idx):
        for epoch in range(50):
            total_loss = 0
            for sentence in data:
                words = sentence.split()
                h_prev = [0] * self._hidden_size

                for i in range(len(words) - 1):
                    x = self.one_hot(word_to_idx[words[i]], self._vocab_size)
                    target_idx = word_to_idx[words[i + 1]]

                    # --- Forward pass ---
                    h_in = self.vec_add(self.matvec_mul(self._Wx, x), self.vec_add(self.matvec_mul(self._Wh, h_prev), self._bh))
                    h_t = self.tanh(h_in)
                    y_raw = self.vec_add(self.matvec_mul(self._Wy, h_t), self._by)
                    y_pred = self.softmax(y_raw)
                    total_loss += self.cross_entropy(y_pred, target_idx)

                    # --- Backward pass ---
                    dy = y_pred[:]
                    dy[target_idx] -= 1  # ∂L/∂y_raw

                    # Update Wy and by
                    for i_v in range(self._vocab_size):
                        for j_h in range(self._hidden_size):
                            self._Wy[i_v][j_h] -= self.learning_rate * dy[i_v] * h_t[j_h]
                        self._by[i_v] -= self.learning_rate * dy[i_v]

                    # Backprop to hidden
                    dh = [sum(self._Wy[i_v][j_h] * dy[i_v] for i_v in range(self._vocab_size)) for j_h in range(self._hidden_size)]
                    dh_raw = [dh[j] * (1 - h_t[j] ** 2) for j in range(self._hidden_size)]

                    for i_h in range(self._hidden_size):
                        for j_v in range(self._vocab_size):
                            self._Wx[i_h][j_v] -= self.learning_rate * dh_raw[i_h] * x[j_v]
                        self._bh[i_h] -= self.learning_rate * dh_raw[i_h]

                    for i_h in range(self._hidden_size):
                        for j_h in range(self._hidden_size):
                            self._Wh[i_h][j_h] -= self.learning_rate * dh_raw[i_h] * h_prev[j_h]

                    h_prev = h_t

            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

    def predict(self, sentence, word_to_idx, idx_to_word):
        h_prev = [0] * self._hidden_size
        words = sentence.split()

        for i in range(len(words) - 1):
            x = self.one_hot(word_to_idx[words[i]], self._vocab_size)
            h_in = self.vec_add(self.matvec_mul(self._Wx, x), self.vec_add(self.matvec_mul(self._Wh, h_prev), self._bh))
            h_t = self.tanh(h_in)
            y_raw = self.vec_add(self.matvec_mul(self._Wy, h_t), self._by)
            y_pred = self.softmax(y_raw)

            predicted_idx = y_pred.index(max(y_pred))
            predicted_word = idx_to_word[predicted_idx]
            print(f"Input: '{words[i]}' → Predicted: '{predicted_word}' (Target: '{words[i+1]}')")

            h_prev = h_t
