import os
from collections import Counter
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()


flags = tf.app.flags

flags.DEFINE_string('train_file', 'oliver.txt', 'text file to train LSTM')
flags.DEFINE_integer('seq_size', 32, 'sequence length')
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_integer('embedding_size', 128, 'Embedding hidden size')
flags.DEFINE_integer('lstm_size', 128, 'LSTM hidden size')
flags.DEFINE_float('dropout_keep_prob', 0.7, 'LSTM dropout keep probability')
flags.DEFINE_integer('gradients_norm', 5, 'norm to clip gradients')
flags.DEFINE_multi_string(
    'initial_words', ['I', 'am'], 'Initial words to start prediction from')
flags.DEFINE_integer('predict_top_k', 5, 'top k results to sample word from')
flags.DEFINE_integer('num_epochs', 20, 'Number of epochs to train')
flags.DEFINE_string('checkpoint_path', 'checkpoint',
                    'directory to store trained weights')

FLAGS = flags.FLAGS


def get_data_from_file(train_file, batch_size, seq_size):
    with open(train_file, encoding='utf-8') as f:
        text = f.read()

    text = text.split()

    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)

    print('Vocabulary size', n_vocab)

    int_text = [vocab_to_int[w] for w in text]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text


def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]


class RNNModule(tf.keras.Model):
    def __init__(self, n_vocab, embedding_size, lstm_size):
        super(RNNModule, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(
            n_vocab, embedding_size)
        self.lstm = tf.keras.layers.CuDNNLSTM(
            lstm_size, return_state=True, return_sequences=True)
        self.dense = tf.keras.layers.Dense(n_vocab)

    def call(self, x, prev_state):
        embed = self.embedding(x)
        output, state_h, state_c = self.lstm(embed, prev_state)
        logits = self.dense(output)
        preds = tf.nn.softmax(logits)
        return logits, preds, (state_h, state_c)

    def zero_state(self, batch_size):
        return [tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size])]


def predict(model, vocab_to_int, int_to_vocab, n_vocab):
    def get_word(int_pred, n_vocab):
        p = np.squeeze(int_pred)
        p[p.argsort()][:-5] = 0
        p = p / np.sum(p)
        word = np.random.choice(n_vocab, 1, p=p)[0]

        return word

    val_state = model.zero_state(1)
    words = ['I', 'am']
    for word in words:
        int_word = tf.convert_to_tensor(
            [[vocab_to_int[word]]], dtype=tf.float32)
        _, int_pred, val_state = model(int_word, val_state)
    int_pred = int_pred.numpy()
    int_word = get_word(int_pred, n_vocab)
    words.append(int_to_vocab[int_word])
    for _ in range(100):
        int_word = tf.convert_to_tensor(
            [[int_word]], dtype=tf.float32)
        _, int_pred, val_state = model(int_word, val_state)
        int_pred = int_pred.numpy()
        int_word = get_word(int_pred, n_vocab)
        words.append(int_to_vocab[int_word])
    print(' '.join(words))


def main(_):
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = \
        get_data_from_file(FLAGS.train_file, FLAGS.batch_size, FLAGS.seq_size)

    model = RNNModule(n_vocab, FLAGS.embedding_size, FLAGS.lstm_size)

    state = model.zero_state(FLAGS.batch_size)

    optimizer = tf.train.AdamOptimizer()

    iteration = 0
    for e in range(FLAGS.num_epochs):
        state = model.zero_state(FLAGS.batch_size)
        batches = get_batches(
            in_text, out_text, FLAGS.batch_size, FLAGS.seq_size)

        for x, y in batches:
            iteration += 1

            x = tf.convert_to_tensor(x, dtype=tf.float32)

            with tf.GradientTape() as tape:
                logits, _, state = model(x, state)
                loss = tf.losses.sparse_softmax_cross_entropy(y, logits)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(e, FLAGS.num_epochs),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss.numpy()))

            if iteration % 1000 == 0:
                predict(model, vocab_to_int, int_to_vocab, n_vocab)


if __name__ == '__main__':
    tf.app.run()
