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

flags = tf.app.flags
flags.DEFINE_integer('window_size', 5, 'window size')
flags.DEFINE_integer('batch_size', 512, 'batch size')
flags.DEFINE_integer('embedding_size', 300, 'embedding size')
flags.DEFINE_integer('num_sampled', 100, 'number of negative samples for NSL computation')
flags.DEFINE_integer('num_iterations', 50000, 'number of iterations for training')
flags.DEFINE_integer('test_size', 16, 'window size')
flags.DEFINE_integer('test_window', 100, 'window size')

FLAGS = flags.FLAGS

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

    # Subsampling makes it worse for eliminating contextual info
    # return train_words, int_to_vocab, vocab_to_int, n_vocab
    return int_words, int_to_vocab, vocab_to_int, n_vocab


def _create_target(batch):
    x = []
    y = []
    for i in range(FLAGS.window_size, len(batch) - FLAGS.window_size):
        x.append(np.append(batch[i - FLAGS.window_size:i],
                           batch[i + 1:i + FLAGS.window_size + 1]))
        y.append(batch[i])
    return np.array(x), np.array(y)[:, None]


def create_batches(int_words, batch_size, window_size=5):
    num_batches = int(len(int_words) // batch_size)
    int_words = int_words[:num_batches * batch_size]

    for i in range(0, len(int_words), batch_size):
        x, y = create_target(int_words[i:i + batch_size], window_size)
        yield x, y


def create_dataset(int_words, batch_size, window_size=5):
    # TODO: need to find a solution for losing window_size target words per two batches
    # The code below can solve that problem, but that takes too long
    # Or, running it once than save the processed sequence somewhere might be an option
    # for i in range(window_size, len(int_words), batch_size - window_size):
    #     print(i / len(int_words))
    #     if i == window_size:
    #         new_int_words = int_words[i - window_size:i - window_size + batch_size]
    #     else:
    #         new_int_words = np.append(
    #                 new_int_words,
    #                 int_words[i - window_size:i - window_size + batch_size])
    num_batches = int(len(int_words) // batch_size)
    int_words = int_words[:num_batches * batch_size]
    int_words = np.reshape(int_words, (-1, batch_size))
    dataset = tf.data.Dataset.from_tensor_slices(int_words)
    dataset = dataset.map(lambda batch: tuple(tf.py_func(
        _create_target, [batch], [tf.int64, tf.int64])))
    iterator = dataset.repeat().make_one_shot_iterator()
    return iterator.get_next()


def get_embed(n_vocab, inputs, embedding_size):
    # Inputs of CBOW will have shape [batch_size, 2 * window_size]
    embedding = tf.get_variable(
        'embedding_weights', [n_vocab, embedding_size],
        initializer=tf.initializers.random_uniform(-1, 1))
    embed = None
    for i in range(2 * FLAGS.window_size):
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


train_words, int_to_vocab, vocab_to_int, n_vocab = read_data_from_file(
    'data/text8')
inputs_, labels_ = create_dataset(train_words, FLAGS.batch_size, FLAGS.window_size)

embedding, embed = get_embed(n_vocab, inputs_, FLAGS.embedding_size)
loss_op, train_op = get_loss_and_train_op(
    n_vocab, embed, FLAGS.embedding_size, labels_, FLAGS.num_sampled)


test_words = np.array(random.sample(range(0, FLAGS.test_window), FLAGS.test_size // 2))
test_words = np.append(test_words, random.sample(range(1000, 1000 + FLAGS.test_window), FLAGS.test_size // 2))
similarity = get_predictions(test_words, embedding)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print_loss = 0
    start = time.time()
    for i in range(FLAGS.num_iterations):
        all_losses = []
        loss, _ = sess.run([loss_op, train_op])
        all_losses.append(loss)
        print_loss += loss
        if i % 100 == 0:
            print('Iteration {}/{}'.format(i, FLAGS.num_iterations),
                  'Average loss {:.4f}'.format(np.mean(print_loss / 100)),
                  'in {:.4f} sec'.format(time.time() - start))
            print_loss = 0
            start = time.time()

        if i % 1000 == 0:
            sims = similarity.eval()
            for ii in range(sims.shape[0]):
                top_k = (-sims[ii, :]).argsort()[:9]
                log = '{}: '.format(int_to_vocab[top_k[0]])
                for k in top_k[1:]:
                    log += '{}, '.format(int_to_vocab[k])
                print(log)

