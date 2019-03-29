# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import unicodedata
import re
import matplotlib.pyplot as plt
import os
import imageio

with open('../data/eng_fra.txt') as f:
    lines = f.read()

raw_data = []
for line in lines.split('\n'):
    raw_data.append(line.split('\t'))

print(raw_data[-5:])
raw_data = raw_data[:-1]


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s


raw_data_en, raw_data_fr = list(zip(*raw_data))
raw_data_en = [normalize_string(data) for data in raw_data_en]
raw_data_fr_in = ['<start> ' + normalize_string(data) for data in raw_data_fr]
raw_data_fr_out = [normalize_string(data) + ' <end>' for data in raw_data_fr]

print(raw_data_fr_out[-5:])

en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
en_tokenizer.fit_on_texts(raw_data_en)
data_en = en_tokenizer.texts_to_sequences(raw_data_en)
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,
                                                        padding='post')
print(data_en[:2])

fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
fr_tokenizer.fit_on_texts(raw_data_fr_in)
fr_tokenizer.fit_on_texts(raw_data_fr_out)
data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in)
data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in,
                                                           padding='post')
print(data_fr_in[:2])

data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out)
data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out,
                                                            padding='post')
print(data_fr_out[:2])

BATCH_SIZE = 64
EMBEDDING_SIZE = 256
RNN_SIZE = 1024

dataset = tf.data.Dataset.from_tensor_slices(
    (data_en, data_fr_in, data_fr_out))
dataset = dataset.shuffle(len(raw_data_en)).batch(
    BATCH_SIZE, drop_remainder=True)


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, rnn_size):
        super(Encoder, self).__init__()
        self.rnn_size = rnn_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            rnn_size, return_sequences=True, return_state=True)

    def call(self, sequence, states):
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)

        return output, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.rnn_size]),
                tf.zeros([batch_size, self.rnn_size]))


en_vocab_size = len(en_tokenizer.word_index) + 1

encoder = Encoder(en_vocab_size, EMBEDDING_SIZE, RNN_SIZE)


class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size):
        super(LuongAttention, self).__init__()
        self.wa = tf.keras.layers.Dense(rnn_size)

    def call(self, decoder_output, encoder_output):
        # Dot score: h_t (dot) h_s
        score = tf.matmul(decoder_output, self.wa(encoder_output), transpose_b=True)

        # alignment a_t
        alignment = tf.nn.softmax(score, axis=2)

        # context vector c_t is the average sum of encoder output
        context = tf.matmul(alignment, encoder_output)

        return context, alignment


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, rnn_size):
        super(Decoder, self).__init__()
        self.attention = LuongAttention(rnn_size)
        self.rnn_size = rnn_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            rnn_size, return_sequences=True, return_state=True)
        self.wc = tf.keras.layers.Dense(rnn_size, activation='tanh')
        self.ws = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, state, encoder_output):
        embed = self.embedding(sequence)
        lstm_out, state_h, state_c = self.lstm(embed, initial_state=state)
        context, alignment = self.attention(lstm_out, encoder_output)

        # concat context and decoder_output
        lstm_out = tf.concat([tf.squeeze(context, 1), tf.squeeze(lstm_out, 1)], 1)
        lstm_out = self.wc(lstm_out)
        logits = self.ws(lstm_out)

        return logits, state_h, state_c, alignment

    def init_states(self, batch_size):
        return tf.zeros([batch_size, self.lstm_size])


fr_vocab_size = len(fr_tokenizer.word_index) + 1
decoder = Decoder(fr_vocab_size, EMBEDDING_SIZE, RNN_SIZE)


def loss_func(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    loss = crossentropy(targets, logits)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=loss.dtype)

    loss = tf.reduce_mean(loss * mask)

    return loss


optimizer = tf.keras.optimizers.Adam(clipnorm=5.0)


def predict(test_source_text=None):
    if test_source_text is None:
        test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]
    print(test_source_text)
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    print(test_source_seq)

    en_initial_states = encoder.init_states(1)
    en_outputs = encoder(tf.constant(test_source_seq), en_initial_states)

    de_input = tf.constant([[fr_tokenizer.word_index['<start>']]])
    de_state_h, de_state_c = en_outputs[1:]
    out_words = []
    alignments = []

    while True:
        de_output, de_state_h, de_state_c, alignment = decoder(
            de_input, (de_state_h, de_state_c), en_outputs[0])
        de_input = tf.expand_dims(tf.argmax(de_output, -1), 0)
        out_words.append(fr_tokenizer.index_word[de_input.numpy()[0][0]])
        
        alignments.append(alignment.numpy())

        if out_words[-1] == '<end>' or len(out_words) >= 20:
            break

    print(' '.join(out_words))
    return np.array(alignments), test_source_text.split(' '), out_words


@tf.function
def train_step(source_seq, target_seq_in, target_seq_out, en_initial_states):
    loss = 0
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq, en_initial_states)
        en_states = en_outputs[1:]
        de_state_h, de_state_c = en_states

        for i in range(target_seq_out.shape[1]):
            decoder_in = tf.expand_dims(target_seq_in[:, i], 1)
            logit, de_state_h, de_state_c, _ = decoder(
                decoder_in, (de_state_h, de_state_c), en_outputs[0])
            loss += loss_func(target_seq_out[:, i], logit)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss / target_seq_out.shape[1]


NUM_EPOCHS = 15


# encoder.load_weights('checkpoints_luong/encoder_15.h5')
# decoder.load_weights('checkpoints_luong/decoder_15.h5')

if not os.path.exists('checkpoints_luong'):
    os.makedirs('checkpoints_luong')

for e in range(NUM_EPOCHS):
    en_initial_states = encoder.init_states(BATCH_SIZE)
    encoder.save_weights('checkpoints_luong/encoder_{}.h5'.format(e + 1))
    decoder.save_weights('checkpoints_luong/decoder_{}.h5'.format(e + 1))
    for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
        loss = train_step(source_seq, target_seq_in,
                          target_seq_out, en_initial_states)
        
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(e + 1, batch, loss.numpy()))

    try:
        predict()

        predict("How are you today ?")
    except Exception:
        continue

        
if not os.path.exists('heatmap'):
    os.makedirs('heatmap')

test_sequence = input()
num_tested = 1
filenames = []

while test_sequence != 'q':
    test_sequence = normalize_data(test_sequence)
    alignments, source, prediction = predict(test_sequence)
    attention = np.squeeze(alignments, (1, 2))
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    ax.set_xticklabels([''] + source)
    ax.set_yticklabels([''] + prediction)
    
    filenames.append('heatmap/test_{}.png')
    plt.savefig('heatmap/test_{}.png')
    plt.close()
    
    num_tested += 1

    test_sequence = input()

with imageio.get_writer('translation_heatmaps.gif', mode='I', duration=0.8) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
