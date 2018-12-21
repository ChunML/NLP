import tensorflow as tf
import numpy as np
from collections import Counter
import random
import utils
from os.path import isfile, isdir
from tqdm import tqdm
from urllib.request import urlretrieve
import zipfile

FLAGS = tf.app.flags.FLAGS

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
    threshold = FLAGS.drop_word_threshold

    freqs = {word: count/total_count for word,
             count in each_word_count.items()}
    probs = {word: 1 - np.sqrt(threshold/freqs[word])
             for word in each_word_count}

    train_words = [word for word in int_words if random.random() <
                   (1 - probs[word])]

    print('After subsampling, first 30 words:', train_words[:30])
    print('After subsampling, total words:', len(train_words))

    return train_words, int_to_vocab, vocab_to_int, n_vocab


def sample_eval_data():
    valid_examples = np.array(random.sample(
        range(FLAGS.valid_window), FLAGS.valid_size // 2))
    valid_examples = np.append(
        valid_examples,
        random.sample(range(1000, 1000 + FLAGS.valid_window),
                      FLAGS.valid_size // 2))
    return valid_examples


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


def get_dataset(words, batch_size, window_size=5):
    def _parse_data(batch):
        x, y = [], []
        for i in range(len(batch)):
            batch_x, batch_y = get_target(batch, i, window_size)
            y.extend(batch_y)
            x.extend(batch_x)
        y = np.expand_dims(y, axis=-1)
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


def get_eval_dataset(words):
    dataset = tf.data.Dataset.from_tensor_slices(words)
    iterator = dataset.batch(1).make_one_shot_iterator()
    return iterator.get_next()
