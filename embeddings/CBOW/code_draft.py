import tensorflow as tf
import numpy as np
from collections import Counter
import random
import utils
from os.path import isfile, isdir
from tqdm import tqdm
from urllib.request import urlretrieve
import zipfile
import time

dataset_folder_path = 'data'
dataset_filename = 'text8.zip'
dataset_name = 'Text8 Dataset'


def maybe_download():
    class DLProgress(tqdm):
        last_block = 0

        def hook(self, block_num=1, block_size=1, total_size=None):
            self.total = total_size
            self.update((block_num - self.last_block) * block_size)
            self.last_block = block_num

    if not isfile(dataset_filename):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc=dataset_name) as pbar:
            urlretrieve(
                'http://mattmahoney.net/dc/text8.zip',
                dataset_filename,
                pbar.hook)

    if not isdir(dataset_folder_path):
        with zipfile.ZipFile(dataset_filename) as zip_ref:
            zip_ref.extractall(dataset_folder_path)


def read_data_from_file(data_path):
    maybe_download()
    with open(data_path) as f:
        text = f.read()

    ###########################################################
    # ------------------- Preprocessing -----------------------
    # 1. Tokenize punctuations e.g. period -> <PERIOD>
    # 2. Remove words that show up five times or fewer
    words = utils.preprocess(text)

    # Hmm, let's take a look at the processed data
    print('First 30 words:', words[:30])
    print('Total words:', len(words))
    print('Total unique words:', len(set(words)))

    # Create two dictionaries to convert words to integers
    vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
    n_vocab = len(int_to_vocab)

    # Convert words into integers
    int_words = [vocab_to_int[w] for w in words]

    ###########################################################
    # ------------------- Subsampling -------------------------
    # Some words like "the", "a", "of" etc don't provide much
    # information. So we might want to remove some of them.
    # This results in faster and better result.
    # The probability that a word is discarded is
    # P(w) = 1 - sqrt(1 / frequency(w))
    each_word_count = Counter(int_words)
    total_count = len(int_words)
    threshold = 1e-5  # FLAGS.drop_word_threshold

    freqs = {word: count/total_count for word,
             count in each_word_count.items()}
    probs = {word: 1 - np.sqrt(threshold/freqs[word])
             for word in each_word_count}

    train_words = [word for word in int_words if random.random() <
                   (1 - probs[word])]

    print('After subsampling, first 30 words:', train_words[:30])
    print('After subsampling, total words:', len(train_words))

    return train_words, int_to_vocab, vocab_to_int, n_vocab


def create_target(batch, window_size):
    x = []
    y = []
    for i in range(window_size, len(batch) - window_size):
        x.append(batch[i - window_size:i] +
                 batch[i + 1:i + window_size + 1])
        y.append(batch[i])
    return x, y


def create_batches(int_words, batch_size, window_size=5):
    num_batches = int(len(int_words) // batch_size)
    int_words = int_words[:num_batches * batch_size]

    for i in range(0, len(int_words), batch_size):
        x, y = create_target(int_words[i:i + batch_size], window_size)
        yield np.array(x), np.array(y)


def get_embed(n_vocab, inputs, embedding_size):
    # Inputs of CBOW will have shape [batch_size, 2 * window_size]
    embedding = tf.get_variable(
        'embedding_weights', [n_vocab, embedding_size],
        initializer=tf.initializers.random_uniform(-1, 1))
    embed = None
    for i in range(2 * window_size):
        inp = tf.squeeze(
            tf.slice(inputs, [0, i], [tf.shape(inputs)[0], 1]), axis=1)
        if embed is None:
            embed = tf.nn.embedding_lookup(embedding, inp)
            embed = tf.expand_dims(embed, axis=2)
        else:
            embed_i = tf.expand_dims(
                tf.nn.embedding_lookup(embedding, inp), axis=2)
            embed = tf.concat([embed, embed_i], axis=2)
    mean_embed = tf.reduce_mean(embed, axis=-1, keepdims=False)
    return embedding, mean_embed


def get_loss_and_train_op(n_vocab, embed, embedding_size, labels, num_sampled):
    with tf.variable_scope('sampled_loss', reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(
            'loss_weights', [n_vocab, embedding_size],
            initializer=tf.initializers.truncated_normal(stddev=0.1))
        biases = tf.get_variable(
            'loss_biases', [n_vocab], initializer=tf.initializers.zeros())
        losses = tf.nn.sampled_softmax_loss(
            weights=weights,
            biases=biases,
            labels=labels,
            inputs=embed,
            num_sampled=num_sampled,
            num_classes=n_vocab)
        loss = tf.reduce_mean(losses)

    train_op = tf.train.AdamOptimizer().minimize(loss)
    return loss, train_op


def get_predictions(test_words, embedding):
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    valid_embed = tf.nn.embedding_lookup(normalized_embedding, test_words)
    similarity = tf.matmul(valid_embed, tf.transpose(normalized_embedding))
    return similarity


batch_size = 1000
window_size = 5
embedding_size = 300
num_sampled = 100
train_words, int_to_vocab, vocab_to_int, n_vocab = read_data_from_file(
    'data/text8')
inputs_ = tf.placeholder(tf.int32, [None, 2 * window_size])
labels_ = tf.placeholder(tf.int32, [None, 1])
embedding, embed = get_embed(n_vocab, inputs_, embedding_size)
loss_op, train_op = get_loss_and_train_op(
    n_vocab, embed, embedding_size, labels_, num_sampled)

test_words = np.random.randint(0, 100, 8)
similarity = get_predictions(test_words, embedding)

num_epochs = 10

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(num_epochs):
        batches = create_batches(train_words, batch_size, window_size)
        batch_loss = []
        iteration = 0
        start = time.time()
        for inputs, labels in batches:
            loss, _ = sess.run(
                [loss_op, train_op],
                feed_dict={
                    inputs_: inputs,
                    labels_: labels[:, None]})
            batch_loss.append(loss)
            if iteration % 100 == 0:
                print('Epoch {}/{}'.format(e, num_epochs),
                      'Iteration {}'.format(iteration),
                      '{:.4f} sec'.format(time.time() - start),
                      'Batch loss {:.4f}'.format(np.mean(batch_loss)))
                start = time.time()
            
            if iteration % 1000 == 0:
                sims = similarity.eval()
                for i in range(sims.shape[0]):
                    top_k = (-sims[i, :]).argsort()[:9]
                    log = '{}: '.format(int_to_vocab[top_k[0]])
                    for k in top_k[1:]:
                        log += '{}, '.format(int_to_vocab[k])
                    print(log)
                batch_loss = []

            iteration += 1
