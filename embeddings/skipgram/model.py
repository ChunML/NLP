import tensorflow as tf
import utils
import numpy as np
from prepare_data import read_data_from_file, get_dataset, get_eval_dataset, sample_eval_data
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

flags.DEFINE_integer('evaluate_every', 1000,
                     'Evaluate the model every ... iterations.')
flags.DEFINE_integer('total_iterations', 50000,
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


def get_embed(n_vocab, inputs):
    embedding = tf.Variable(tf.random_uniform(
        [n_vocab, FLAGS.embedding_size], -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs)
    return embedding, embed


def get_loss_and_training_op(n_vocab, labels, embed):
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


def infer(predictions, int_to_vocab):
    for prediction in predictions:
        sim = prediction['similarity']
        top_10_words = sim.argsort()[-11:]
        words = [int_to_vocab[w] for w in top_10_words]
        print('Words nearest to {}:'.format(
            words[-1]), ' '.join(words[:-1]))


def model_fn(features, labels, mode, params):
    n_vocab = params['n_vocab']
    embedding, embed = get_embed(n_vocab, features)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss, train_op = get_loss_and_training_op(n_vocab, labels, embed)
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


def create_estimator(n_vocab):
    config = tf.estimator.RunConfig(
        model_dir='estimator_checkpoint',
        save_checkpoints_steps=5000
    )
    return tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params={
            'n_vocab': n_vocab
        })


def main(unused_argv):
    train_words, int_to_vocab, vocab_to_int, n_vocab = read_data_from_file(
        'data/text8')
    estimator = create_estimator(n_vocab)
    if FLAGS.mode == 'train':
        valid_words = sample_eval_data()
        for _ in range(int(FLAGS.total_iterations / FLAGS.evaluate_every)):
            estimator.train(
                input_fn=lambda: get_dataset(
                    train_words, FLAGS.batch_size, FLAGS.window_size),
                steps=FLAGS.evaluate_every)

            predictions = estimator.predict(
                input_fn=lambda: get_eval_dataset(valid_words))
            infer(predictions, int_to_vocab)

    elif FLAGS.mode == 'predict':
        test_words = vocab_to_int[FLAGS.test_word]
        predictions = estimator.predict(
            input_fn=lambda: get_eval_dataset([test_words]))
        infer(predictions, int_to_vocab)


if __name__ == '__main__':
    tf.app.run()
