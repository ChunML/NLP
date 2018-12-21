import tensorflow as tf
import utils
from collections import Counter
import numpy as np
import random
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags

flags.DEFINE_string('mode', 'train',
                    'Running mode.')
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


with open('data/text8') as f:
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

freqs = {word: count/total_count for word, count in each_word_count.items()}
probs = {word: 1 - np.sqrt(threshold/freqs[word]) for word in each_word_count}

train_words = [word for word in int_words if random.random() <
               (1 - probs[word])]

print('After subsampling, first 30 words:', train_words[:30])
print('After subsampling, total words:', len(train_words))

valid_examples = np.array(random.sample(
    range(FLAGS.valid_window), FLAGS.valid_size // 2))
valid_examples = np.append(valid_examples,
                           random.sample(range(1000, 1000 + FLAGS.valid_window),
                                         FLAGS.valid_size // 2))

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


def get_embed(inputs):
    embedding = tf.Variable(tf.random_uniform(
        [n_vocab, FLAGS.embedding_size], -1, 1))
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
    optimizer = tf.train.AdamOptimizer().minimize(
        cost, global_step=tf.train.get_global_step())
    return cost, optimizer


def model_fn(features, labels, mode):
    embedding, embed = get_embed(features)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss, train_op = get_loss_and_training_op(labels, embed)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.PREDICT:
        norm = tf.sqrt(tf.reduce_sum(
            tf.square(embedding), 1, keepdims=True))
        normalized_embedding = embedding / norm
        valid_embedding = tf.nn.embedding_lookup(
            normalized_embedding, features)
        similarity = tf.matmul(valid_embedding,
                               tf.transpose(normalized_embedding))
        predictions = {'similarity': similarity}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


def create_estimator():
    config = tf.estimator.RunConfig(
        model_dir='estimator_checkpoint',
        save_checkpoints_steps=5000
    )
    return tf.estimator.Estimator(
        model_fn=model_fn,
        config=config)


def main(unused_argv):
    estimator = create_estimator()
    if FLAGS.mode == 'train':
        estimator.train(
            input_fn=lambda: get_dataset(
                train_words, FLAGS.batch_size, FLAGS.window_size),
            max_steps=FLAGS.num_iterations)
    elif FLAGS.mode == 'eval':
        predictions = estimator.predict(
            input_fn=lambda: get_eval_dataset(valid_examples))
        for prediction in predictions:
            print(int_to_vocab[prediction['similarity'].argsort()[-1]])


if __name__ == '__main__':
    tf.app.run()
