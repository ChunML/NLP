import tensorflow as tf
import utils
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS


def get_embed(n_vocab, inputs):
    embedding = tf.Variable(tf.random_uniform(
        [n_vocab, FLAGS.embedding_size], -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs)
    return embedding, embed


def get_loss_and_training_op(n_vocab, labels, embed):
    embed.set_shape([None, FLAGS.embedding_size])
    dense = tf.layers.dense(
        embed, FLAGS.hidden_size,
        kernel_initializer=tf.initializers.truncated_normal(stddev=0.1))
    weights = tf.Variable(tf.truncated_normal(
        [n_vocab, FLAGS.hidden_size], stddev=0.1))
    biases = tf.Variable(tf.zeros(n_vocab))

    loss = tf.nn.sampled_softmax_loss(weights=weights,
                                      biases=biases,
                                      labels=labels,
                                      inputs=dense,
                                      num_sampled=FLAGS.n_sampled,
                                      num_classes=n_vocab)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(
        cost, global_step=tf.train.get_global_step())
    return cost, optimizer


def get_top_10_words(predictions, int_to_vocab):
    all_words = []
    for prediction in predictions:
        sim = prediction['similarity']
        top_10_words = sim.argsort()[-11:]
        words = [int_to_vocab[w] for w in top_10_words]
        print('Words nearest to {}:'.format(
            words[-1]), ' '.join(words[:-1]))
        all_words.append(words)
    return all_words


def model_fn(features, labels, mode, params):
    n_vocab = params['n_vocab']
    embedding, embed = get_embed(n_vocab, features)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss, train_op = get_loss_and_training_op(n_vocab, labels, embed)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        raise NotImplementedError
    elif mode == tf.estimator.ModeKeys.PREDICT:
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
            'n_vocab': n_vocab})
