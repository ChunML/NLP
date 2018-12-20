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
#------------------- Preprocessing ------------------------
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
#------------------- Subsampling --------------------------
# Some words like "the", "a", "of" etc don't provide much
# information. So we might want to remove some of them.
# This results in faster and better result.
# The probability that a word is discarded is
# P(w) = 1 - sqrt(1 / frequency(w))
each_word_count = Counter(int_words)
total_count = len(int_words)
threshold = 1e-5

freqs = {word: count/total_count for word, count in each_word_count.items()}
probs = {word: 1 - np.sqrt(threshold/freqs[word]) for word in each_word_count}

train_words = [word for word in int_words if random.random() < (1 - probs[word])]

print('After subsampling, first 30 words:', train_words[:30])
print('After subsampling, total words:', len(train_words))

###########################################################
#------------------- Making Batch -------------------------
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
    return target_words

# Then we define a function to create batches for the data
# Each bactch contains batch_size words, and we use get_target
# above to get the target words for each
# Use generator to make it efficient
def get_batches(words, batch_size, window_size=5):
    n_batches = int(len(words) / batch_size)
    words = words[:n_batches * batch_size]

    for i in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[i: i + batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x] * len(batch_y))
        yield x, y

###########################################################
#------------------- Build Graph --------------------------
# Input to network will be an array of length batch_size
# To make things work, labels must have second dimensions set to None or 1
# Embedding layer will map input of [batch_size, n_vocab] to [batch_size, hidden_size]
# Initialize embedding weights with random uniform (-1, 1)
# For the loss, because n_vocab is very large, updating all the weights
# will be very costly. So we randomly choose some negative labels
# to compute the loss, and update the network.
# This is called Negative Sampling.
n_vocab = len(int_to_vocab)
n_embedding = 300
n_sampled = 100

# train_graph = tf.Graph()
# with train_graph.as_default():

inputs = tf.placeholder(tf.int32, [None])
labels = tf.placeholder(tf.int32, [None, None])

def get_embed(inputs):
    embedding = tf.Variable(tf.random_uniform([n_vocab, n_embedding], -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs)
    return embedding, embed

embedding, embed = get_embed(inputs)

def get_loss_and_training_op(labels, embed):
    weights = tf.Variable(tf.truncated_normal([n_vocab, n_embedding], stddev=0.1))
    biases = tf.Variable(tf.zeros(n_vocab))
    # logits = tf.layers.dense(embed, n_vocab)

    loss = tf.nn.sampled_softmax_loss(weights=weights,
                                      biases=biases,
                                      labels=labels,
                                      inputs=embed,
                                      num_sampled=n_sampled,
                                      num_classes=n_vocab)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    return cost, optimizer

cost, optimizer = get_loss_and_training_op(labels, embed)

# Below is some code to validate the training process.
# We choose some common words and uncommon words.
# Then we print out closest words to them
# to check if embedding layer is learning well
# with train_graph.as_default():
valid_size = 16
valid_window = 100
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
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)

    # Let's see some magic here:
    # input (N, vocab_size) * embedding (vocab_size, hidden_size) => embed (N, hidden_size)
    # embed (N, hidden_size) * embedding.T (hidden_size, vocab_size) => (N, vocab_size)
    # That's a matrix with same shape of the input
    # So, matmul here is something like a reverse embedding
    # If the embedding layer learned well, the two matrix should be similar
    similarity = tf.matmul(valid_embedding,
                           tf.transpose(normalized_embedding))
    return similarity

valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
valid_examples = np.append(valid_examples,
                           random.sample(range(1000, 1000 + valid_window),
                                         valid_size // 2))

similarity = inference(valid_examples, embedding)

test_word = 'rose'
test_word_int = np.array([vocab_to_int[test_word]])
synonym = inference(test_word_int, embedding)

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
epochs = 10
batch_size = 1000
window_size = 10

checkpoint_dir = 'checkpoint'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

with tf.Session() as sess:
    saver = tf.train.Saver()
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(epochs):
        batches = get_batches(train_words, batch_size, window_size)
        start = time.time()
        for x, y in batches:
            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer],
                                     feed_dict=feed)
            loss += train_loss

            if iteration % 100 == 0:
                end = time.time()
                print('Epoch {}/{}'.format(e, epochs),
                      'Iteration {}'.format(iteration),
                      'Avg Training loss: {:.4f}'.format(loss / 100),
                      '{:.4f} sec/batch'.format((end - start) / 100))
                loss = 0
                start = time.time()
            
            if iteration % 1000 == 0:
                sim = similarity.eval()
                print_inference_result(valid_examples, sim)

                sym = synonym.eval()
                print_inference_result(test_word_int, sym)

            iteration += 1

    checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
    saver.save(sess, 'model.ckpt')
