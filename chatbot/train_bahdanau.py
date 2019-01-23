from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import lookup_ops
import codecs
import numpy as np

# TODO: Use tf.app.flags
flags = tf.app.flags
flags.DEFINE_string('source_data_file', 'processed_input_data.txt', 'path to source data')
flags.DEFINE_string('target_data_file', 'processed_target_data.txt', 'path to target data')
flags.DEFINE_string('vocab_file', 'vocab.txt', 'path to source vocab file')
flags.DEFINE_string('unk', '<unk>', 'unknown token for not-in-vocabulary word')
flags.DEFINE_string('sos', '<sos>', 'start-of-sentence token')
flags.DEFINE_string('eos', '<eos>', 'end-of-sentence token')
flags.DEFINE_integer('unk_id', 0, 'index of unknown token')
flags.DEFINE_integer('hidden_size', 300, 'hidden size of RNN cell')
flags.DEFINE_integer('encoder_num_layers', 2, 'number of layers of encoder')
flags.DEFINE_integer('decoder_num_layers', 2, 'number of layers of decoder')
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_float('keep_prob', 0.8, 'keeping ratio for dropout')
flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate')
flags.DEFINE_integer('source_max_length', 20, 'maximum length of source sequence')
flags.DEFINE_integer('target_max_length', 20, 'maximum length of target sequence')
flags.DEFINE_float('max_gradient', 5.0, 'threshold value for gradient clipping')
flags.DEFINE_integer('num_iterations', 17000, 'number of iterations for training')
flags.DEFINE_integer('print_every', 100, 'print loss and sample every ... iterations')
flags.DEFINE_integer('save_every', 1000, 'save checkpoint every ... iterations')

FLAGS = flags.FLAGS

# ======================== DATA READING =============================
def load_vocab(vocab_file):
  vocab = []
  with open(vocab_file, 'r') as f:
    vocab_size = 0
    for word in f:
      vocab.append(word.strip())
      vocab_size += 1
  return vocab, vocab_size

def create_input_data(source_data_file, target_data_file,
                      vocab_file,
                      batch_size, sos, eos, unk_id,
                      source_max_length, target_max_length):
  source_dataset = tf.data.TextLineDataset(tf.gfile.Glob(source_data_file))
  target_dataset = tf.data.TextLineDataset(tf.gfile.Glob(target_data_file))
  vocab = lookup_ops.index_table_from_file(vocab_file, default_value=unk_id)

  output_buffer_size = batch_size * 1000

  sos_id = tf.cast(vocab.lookup(tf.constant(sos)), tf.int32)
  eos_id = tf.cast(vocab.lookup(tf.constant(eos)), tf.int32)

  dataset = tf.data.Dataset.zip((source_dataset, target_dataset))
  dataset = dataset.map(
    lambda src, tgt: (tf.string_split([src]).values,
                      tf.string_split([tgt]).values)).prefetch(output_buffer_size)
  dataset = dataset.filter(
    lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))
  # dataset = dataset.map(
  #   lambda src, tgt: (src[:source_max_length], tgt[:target_max_length]))
  dataset = dataset.filter(
    lambda src, tgt: tf.logical_and(tf.size(src) <= source_max_length, tf.size(tgt) <= target_max_length))
  dataset = dataset.prefetch(output_buffer_size)

  dataset = dataset.map(
    lambda src, tgt: (tf.cast(vocab.lookup(src), tf.int32),
                      tf.cast(vocab.lookup(tgt), tf.int32)))
  dataset = dataset.prefetch(output_buffer_size)

  dataset = dataset.map(
    lambda src, tgt: (src,
                      tf.concat(([sos_id], tgt), 0),
                      tf.concat((tgt, [eos_id]), 0))).prefetch(output_buffer_size)

  dataset = dataset.map(
    lambda src, tgt_in, tgt_out: (
      src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in))).prefetch(output_buffer_size)

  dataset = dataset.shuffle(100).repeat().padded_batch(
    batch_size,
    padded_shapes=(tf.TensorShape([None]),
                   tf.TensorShape([None]),
                   tf.TensorShape([None]),
                   tf.TensorShape([]),
                   tf.TensorShape([])),
    padding_values=(eos_id,
                    eos_id,
                    eos_id,
                    0,
                    0))

  iterator = dataset.make_initializable_iterator()

  return iterator.get_next(), iterator.initializer, vocab

# ======================== SEQ2SEQ NETWORK =============================
def create_network(source_sequence, sos, eos,
                   target_sequence_in, target_sequence_out,
                   vocab,
                   source_sequence_length,
                   target_sequence_length,
                   vocab_size,
                   hidden_size, keep_prob, batch_size,
                   encoder_num_layers, decoder_num_layers):
  with tf.variable_scope('encoder'):
    encoder_embedding = tf.get_variable(
      'encoder_embedding_weights',
      [vocab_size, hidden_size],
      dtype=tf.float32,
      initializer=tf.initializers.random_uniform(-1, 1, dtype=tf.float32))

    source_sequence = tf.transpose(source_sequence)
    encoder_embedded = tf.nn.embedding_lookup(encoder_embedding, source_sequence)

    # TODO: Update to bidirectional RNN
    def _create_encoder_cell(hidden_size):
      cell =  tf.nn.rnn_cell.LSTMCell(hidden_size)
      return tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)

    bi_num_layers = int(encoder_num_layers / 2)

    if bi_num_layers == 1:
      fw_encoder_lstm = _create_encoder_cell(hidden_size)
      bw_encoder_lstm = _create_encoder_cell(hidden_size)
    else:
      fw_encoder_lstm = tf.nn.rnn_cell.MultiRNNCell(
        [_create_encoder_cell(hidden_size) for _ in range(bi_num_layers)])
      bw_encoder_lstm = tf.nn.rnn_cell.MultiRNNCell(
        [_create_encoder_cell(hidden_size) for _ in range(bi_num_layers)])

    encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
      fw_encoder_lstm,
      bw_encoder_lstm,
      encoder_embedded,
      dtype=tf.float32,
      time_major=True,
      sequence_length=source_sequence_length)

    if bi_num_layers == 1:
      encoder_state = bi_encoder_state
    else:
      encoder_state = []
      for layer_i in range(bi_num_layers):
        encoder_state.append(bi_encoder_state[0][layer_i])
        encoder_state.append(bi_encoder_state[1][layer_i])
      encoder_state = tuple(encoder_state)

    encoder_outputs = tf.concat(encoder_outputs, -1)

  with tf.variable_scope('attention'):
    attention_state = tf.transpose(encoder_outputs, [1, 0, 2])
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
      hidden_size,
      attention_state,
      memory_sequence_length=source_sequence_length)

  with tf.variable_scope('decoder'):
    decoder_embedding = tf.get_variable(
      'decoder_embedding_weights',
      [vocab_size, hidden_size],
      dtype=tf.float32,
      initializer=tf.initializers.random_uniform(-1, 1, dtype=tf.float32))
    target_sequence_in = tf.transpose(target_sequence_in)
    decoder_embedded = tf.nn.embedding_lookup(decoder_embedding, target_sequence_in)

    # TODO: Update to bidirectional RNN
    def _create_decoder_cell(hidden_size):
      cell =  tf.nn.rnn_cell.LSTMCell(hidden_size)
      return tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
    decoder_lstm = tf.nn.rnn_cell.MultiRNNCell(
      [_create_decoder_cell(hidden_size) for _ in range(decoder_num_layers)])
    decoder_lstm = tf.contrib.seq2seq.AttentionWrapper(
      decoder_lstm, attention_mechanism,
      attention_layer_size=hidden_size)
    decoder_output_layer = tf.layers.Dense(vocab_size, use_bias=False)

    decoder_initial_state = decoder_lstm.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

    helper = tf.contrib.seq2seq.TrainingHelper(
      decoder_embedded,
      target_sequence_length,
      time_major=True)

    my_decoder = tf.contrib.seq2seq.BasicDecoder(
      decoder_lstm,
      helper,
      decoder_initial_state,
      decoder_output_layer)

    decoder_outputs, decoder_final_states, _ = tf.contrib.seq2seq.dynamic_decode(
      my_decoder,
      output_time_major=True,
      swap_memory=True)

    target_sequence_out = tf.transpose(target_sequence_out)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=target_sequence_out, logits=decoder_outputs.rnn_output)
    loss_weights = tf.sequence_mask(
      target_sequence_length, tf.shape(target_sequence_out)[0],
      dtype=tf.float32)
    loss_weights = tf.transpose(loss_weights)

    loss = tf.reduce_sum(cross_entropy * loss_weights) / tf.to_float(batch_size)

  with tf.variable_scope('decoder', reuse=True):
    target_sos_id = tf.cast(vocab.lookup(tf.constant(sos)), tf.int32)
    target_eos_id = tf.cast(vocab.lookup(tf.constant(eos)), tf.int32)
    infer_sequence_in = tf.fill([batch_size], target_sos_id)

    infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
      decoder_embedding,
      infer_sequence_in,
      target_eos_id)
    infer_decoder = tf.contrib.seq2seq.BasicDecoder(
      decoder_lstm,
      infer_helper,
      decoder_initial_state,
      decoder_output_layer)

    maximum_iterations = tf.round(tf.reduce_max(source_sequence_length) * 2)
    infer_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
      infer_decoder,
      maximum_iterations=maximum_iterations,
      output_time_major=True,
      swap_memory=True)
    preds = infer_decoder_outputs.sample_id
  return loss, source_sequence, target_sequence_in, preds

def create_train_op(loss, max_gradient, learning_rate):
  # global_step = tf.Variable(0, trainable=False)
  global_step = tf.train.get_or_create_global_step()

  params = tf.trainable_variables()
  gradients = tf.gradients(
    loss,
    params)
  clipped_gradients, _ = tf.clip_by_global_norm(
    gradients, max_gradient)

  # TODO: Schedule learning_rate update
  opt = tf.train.AdamOptimizer(learning_rate)
  train_op = opt.apply_gradients(
    zip(clipped_gradients, params), global_step)
  return global_step, train_op

int_to_vocab, vocab_size = load_vocab(FLAGS.vocab_file)

(source_sequence,
 target_sequence_in, target_sequence_out,
 source_sequence_length, target_sequence_length),\
 iterator_initializer, vocab = create_input_data(
  FLAGS.source_data_file, FLAGS.target_data_file,
  FLAGS.vocab_file,
  FLAGS.batch_size, FLAGS.sos, FLAGS.eos, FLAGS.unk_id,
  FLAGS.source_max_length, FLAGS.target_max_length)

loss, t_source_sequence, t_target_sequence_in, preds = create_network(
  source_sequence, FLAGS.sos, FLAGS.eos,
  target_sequence_in, target_sequence_out,
  vocab,
  source_sequence_length,
  target_sequence_length,
  vocab_size,
  FLAGS.hidden_size, FLAGS.keep_prob, FLAGS.batch_size,
  FLAGS.encoder_num_layers, FLAGS.decoder_num_layers)

global_step, train_op = create_train_op(loss, FLAGS.max_gradient,
                                        FLAGS.learning_rate)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
sess.run(iterator_initializer)

saver = tf.train.Saver()
latest_checkpoint = tf.train.latest_checkpoint('checkpoint_bahdanau')
if latest_checkpoint and tf.train.checkpoint_exists(latest_checkpoint):
  saver.restore(sess, latest_checkpoint)

for _ in range(FLAGS.num_iterations):
  i = global_step.eval(sess)
  if i >= FLAGS.num_iterations:
    print('Training complete!')
    break
  src_seq, tar_seq, predictions, loss_value, _ = sess.run(
    [t_source_sequence, t_target_sequence_in, preds, loss, train_op])
  if (i + 1) % FLAGS.print_every == 0:
    random_id = np.random.choice(src_seq.shape[1])
    print('Step {}: loss {:.4f}'.format(i + 1, loss_value))
    src_sent = ' '.join([int_to_vocab[ix] for ix in src_seq[:, random_id]])
    tar_sent = ' '.join([int_to_vocab[ix] for ix in tar_seq[:, random_id]])
    pred_sent = ' '.join([int_to_vocab[ix] for ix in predictions[:, random_id]])

    if FLAGS.eos in src_sent:
      eos_index = src_sent.index(FLAGS.eos)
      src_sent = src_sent[:eos_index]
    if FLAGS.eos in tar_sent:
      eos_index = tar_sent.index(FLAGS.eos)
      tar_sent = tar_sent[:eos_index]
    if FLAGS.eos in pred_sent:
      eos_index = pred_sent.index(FLAGS.eos)
      pred_sent = pred_sent[:eos_index]
    print('<src>', src_sent.encode('utf-8'))
    print('<dst>', tar_sent.encode('utf-8'))
    print('<pred>', pred_sent.encode('utf-8'))
    print()

  if (i + 1) % FLAGS.save_every == 0:
    print('Saving checkpoint for step {}...\n'.format(i + 1))
    saver.save(sess, 'checkpoint_bahdanau/model-{}.ckpt'.format(i + 1))
