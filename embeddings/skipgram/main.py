import tensorflow as tf
from prepare_data import read_data_from_file, get_dataset, get_eval_dataset, sample_eval_data
from model import create_estimator, infer

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
