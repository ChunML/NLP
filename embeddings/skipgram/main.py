import time
import numpy as np
import tensorflow as tf
import utils

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile
import random
from collections import Counter
import os

flags = tf.app.flags

flags.DEFINE_float('drop_word_threshold', 1e-5,
                   'Threshold to compute probability to drop words in sequence.')
flags.DEFINE_integer('embedding_size', 300,
                     'Embedding layer\' hidden size.')
flags.DEFINE_integer('n_sampled', 100,
                     'Number of negative samples to compute loss.')
flags.DEFINE_integer('valid_size', 16,
                     'Number of words to perform inference.')
flags.DEFINE_integer('valid_window', 100,
                     'Number of words to randomly get sample from.')
flags.DEFINE_string('test_word', 'japan',
                    'Specific word of user choice to perform inference.')

flags.DEFINE_integer('num_iterations', 50000,
                     'Number of training iterations.')
flags.DEFINE_integer('batch_size', 1000,
                     'Batch size used when training.')
flags.DEFINE_integer('window_size', 10,
                     'Window size to compute skip-gram targets.')
flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                    'Checkpoint directory.')
flags.DEFINE_integer('print_every', 100,
                     'Print loss every ... iterations.')
flags.DEFINE_integer('infer_every', 1000,
                     'Infer every ... iteration.')

FLAGS = flags.FLAGS

dataset_folder_path = 'data'
dataset_filename = 'text8.zip'
dataset_name = 'Text8 Dataset'


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

with open('data/text8') as f:
    text = f.read()

###########################################################
# ------------------- Preprocessing ------------------------
# 1. Tokenize punctuations e.g. period -> <PERIOD>
# 2. Remove words that show up five times or fewer
words = utils.preprocess(text)

# Hmm, let's take a look at the processed data
print('First 30 words:', words[:30])
print('Total words:', len(words))
print('Total unique words:', len(set(words)))

# Create two dictionaries to convert words to integers
vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)

# Convert words into integers
int_words = [vocab_to_int[w] for w in words]

###########################################################
# ------------------- Subsampling --------------------------
# Some words like "the", "a", "of" etc don't provide much
# information. So we might want to remove some of them.
# This results in faster and better result.
# The probability that a word is discarded is
# P(w) = 1 - sqrt(1 / frequency(w))
each_word_count = Counter(int_words)
total_count = len(int_words)
threshold = FLAGS.drop_word_threshold

freqs = {word: count/total_count for word, count in each_word_count.items()}
probs = {word: 1 - np.sqrt(threshold/freqs[word]) for word in each_word_count}

train_words = [word for word in int_words if random.random() <
               (1 - probs[word])]

print('After subsampling, first 30 words:', train_words[:30])
print('After subsampling, total words:', len(train_words))

###########################################################
# ------------------- Making Batch -------------------------
# For the skip-gram model to work, for each word in the text
# we must grab words in a window around that word, size C
# Ex: if C = 5, then we choose a R randomly in [1, C]
# then pickup R words before and after current word
# to use as training labels


def get_target(words, idx, window_size=5):
    random_window = random.randint(1, window_size)
    target_words = []
    start_idx = max(0, idx - random_window)
    end_idx = min(len(words) - 1, idx + random_window)
    for i in range(start_idx, end_idx + 1):
        if i == idx:
            continue
        target_words.append(words[i])
    return [words[idx]] * len(target_words), target_words

# Then we define a function to create batches for the data
# Each bactch contains batch_size words, and we use get_target
# above to get the target words for each
# Use generator to make it efficient


def get_dataset(words, batch_size, window_size=5):
    def _parse_data(batch):
        x, y = [], []
        for i in range(len(batch)):
            batch_x, batch_y = get_target(batch, i, window_size)
            y.extend(batch_y)
            x.extend(batch_x)
        return x, y

    n_batches = int(len(words) / batch_size)
    words = words[:n_batches * batch_size]
    words = np.reshape(words, [-1, batch_size])
    dataset = tf.data.Dataset.from_tensor_slices(words)
    dataset = dataset.map(lambda batch: tuple(
        tf.py_func(_parse_data, [batch], [tf.int64, tf.int64])))
    dataset = dataset.repeat()

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


###########################################################
# ------------------- Build Graph --------------------------
# Input to network will be an array of length batch_size
# To make things work, labels must have second dimensions set to None or 1
# Embedding layer will map input of [batch_size, n_vocab] to [batch_size, hidden_size]
# Initialize embedding weights with random uniform (-1, 1)
# For the loss, because n_vocab is very large, updating all the weights
# will be very costly. So we randomly choose some negative labels
# to compute the loss, and update the network.
# This is called Negative Sampling.
n_vocab = len(int_to_vocab)


def get_embed(inputs):
    embedding = tf.Variable(tf.random_uniform([n_vocab, FLAGS.embedding_size], -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs)
    return embedding, embed


def get_loss_and_training_op(labels, embed):
    weights = tf.Variable(tf.truncated_normal(
        [n_vocab, FLAGS.embedding_size], stddev=0.1))
    biases = tf.Variable(tf.zeros(n_vocab))

    loss = tf.nn.sampled_softmax_loss(weights=weights,
                                      biases=biases,
                                      labels=labels,
                                      inputs=embed,
                                      num_sampled=FLAGS.n_sampled,
                                      num_classes=n_vocab)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    return cost, optimizer


# Below is some code to validate the training process.
# We choose some common words and uncommon words.
# Then we print out closest words to them
# to check if embedding layer is learning well
# with train_graph.as_default():


def inference(examples, embedding):
    # Since words in int_to_vocab is sorted by frequency,
    # lower index means that word appears more often.
    # Here we peek 4 elements between (1, 100)
    # and 4 elements between (1000, 1100)

    valid_dataset = tf.constant(examples, dtype=tf.int32)

    # Calculate cosine distance
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))

    # Okay, we normalize the embedding weights by dividing it by norm
    normalized_embedding = embedding / norm
    valid_embedding = tf.nn.embedding_lookup(
        normalized_embedding, valid_dataset)

    # Let's see some magic here:
    # input (N, vocab_size) * embedding (vocab_size, hidden_size) => embed (N, hidden_size)
    # embed (N, hidden_size) * embedding.T (hidden_size, vocab_size) => (N, vocab_size)
    # That's a matrix with same shape of the input
    # So, matmul here is something like a reverse embedding
    # If the embedding layer learned well, the two matrix should be similar
    similarity = tf.matmul(valid_embedding,
                           tf.transpose(normalized_embedding))
    return similarity


valid_examples = np.array(random.sample(range(FLAGS.valid_window), FLAGS.valid_size // 2))
valid_examples = np.append(valid_examples,
                           random.sample(range(1000, 1000 + FLAGS.valid_window),
                                         FLAGS.valid_size // 2))
test_word_int = np.array([vocab_to_int[FLAGS.test_word]])


def print_inference_result(input_words, output_words, top_k=8):
    for i in range(len(output_words)):
        valid_word = int_to_vocab[input_words[i]]

        # sim is a matrix with shape [valid_size, n_vocab]
        # We loop through it and take top_k values along second dimensions
        # TODO: do some readings to find out why we need the minus
        nearest = (-output_words[i, :]).argsort()[1:top_k + 1]
        log = 'Nearest to {}:'.format(valid_word)
        for k in range(top_k):
            close_word = int_to_vocab[nearest[k]]
            log = '{} {},'.format(log, close_word)
        print(log)
    print()


# Train the network with setting like below
num_iter = FLAGS.num_iterations
batch_size = FLAGS.batch_size
window_size = FLAGS.window_size

checkpoint_dir = FLAGS.checkpoint_dir
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

inputs, labels = get_dataset(train_words, FLAGS.batch_size, FLAGS.window_size)
labels = tf.expand_dims(labels, axis=-1)
embedding, embed = get_embed(inputs)
cost, optimizer = get_loss_and_training_op(labels, embed)
similarity = inference(valid_examples, embedding)
synonym = inference(test_word_int, embedding)

with tf.Session() as sess:
    saver = tf.train.Saver()
    loss = 0
    sess.run(tf.global_variables_initializer())

    for i in range(1, num_iter + 1):
        start = time.time()

        train_loss, _ = sess.run([cost, optimizer])
        loss += train_loss
        if i % FLAGS.print_every == 0:
            end = time.time()
            print('Iteration {}/{}'.format(i, num_iter),
                  'Avg Training loss: {:.4f}'.format(loss / 100),
                  '{:.4f} sec/batch'.format(end - start))
            loss = 0
            start = time.time()

        if i % FLAGS.infer_every == 0:
            sim = similarity.eval()
            print_inference_result(valid_examples, sim)

            sym = synonym.eval()
            print_inference_result(test_word_int, sym)

    checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path)
