import tensorflow as tf
from prepare_data import read_data_from_file, get_dataset, get_eval_dataset, sample_eval_data
from model import get_embed, get_loss_and_training_op, get_predictions, get_top_10_words
import numpy as np
import time

flags = tf.app.flags

flags.DEFINE_string('mode', 'train',
                    'Running mode.')
flags.DEFINE_float('drop_word_threshold', 1e-5,
                   'Threshold to compute probability to drop words in sequence.')
flags.DEFINE_integer('embedding_size', 300,
                     'Embedding layer\' hidden size.')
flags.DEFINE_integer('hidden_size', 256,
                     'Dense layer\' hidden size.')
flags.DEFINE_integer('n_sampled', 100,
                     'Number of negative samples to compute loss.')
flags.DEFINE_integer('valid_size', 16,
                     'Number of words to perform inference.')
flags.DEFINE_integer('valid_window', 100,
                     'Number of words to randomly get sample from.')
flags.DEFINE_string('test_word', None,
                    'Specific word of user choice to perform inference.')

flags.DEFINE_integer('log_every', 100,
                     'Evaluate the model every ... iterations.')
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


def predict(valid_words, embedding, int_to_vocab):
    pred_op = get_predictions(valid_words, embedding)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore('checkpoint/model.ckpt')
        predictions = sess.run(pred_op)
        words = get_top_10_words(predictions, int_to_vocab)
    return words


def train(n_vocab, labels, embedding, embed, int_to_vocab):
    loss_op, train_op = get_loss_and_training_op(n_vocab, labels, embed)
    valid_words = sample_eval_data()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        losses = []
        sess.run(tf.global_variables_initializer())
        start = time.time()
        for i in range(FLAGS.total_iterations):
            loss, _ = sess.run([loss_op, train_op])
            losses.append(loss)
            if i % FLAGS.log_every == 0:
                end = time.time()
                print('Iteration {}/{} '.format(i, FLAGS.total_iterations),
                      'Average Loss: {:.4f}'.format(np.mean(losses)),
                      '{:.4f} sec/{} iterations'.format((end - start), FLAGS.log_every))
                start = time.time()

            if i % FLAGS.evaluate_every == 0:
                saver.save(sess, 'checkpoint/model-{}.ckpt'.format(i))
                pred_op = get_predictions(valid_words, embedding)
                predictions = sess.run(pred_op)
                words = get_top_10_words(predictions, int_to_vocab)
        saver.save(sess, 'checkpoint/model.ckpt')


def main(unused_argv):
    train_words, int_to_vocab, vocab_to_int, n_vocab = read_data_from_file(
        'data/text8')
    inputs, labels = get_dataset(
        train_words, FLAGS.batch_size, FLAGS.window_size)
    embedding, embed = get_embed(n_vocab, inputs)

    if FLAGS.mode == 'train':
        train(n_vocab, labels, embedding, embed, int_to_vocab)
    if FLAGS.mode == 'predict':
        valid_words = [vocab_to_int[FLAGS.test_word]]
        predict(valid_words, embedding, int_to_vocab)


if __name__ == '__main__':
    tf.app.run()
