from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import lookup_ops
import numpy as np
from process_cornell import process_line

# TODO: Use tf.app.flags
flags = tf.app.flags
flags.DEFINE_string('vocab_file', 'vocab.txt', 'path to source vocab file')
flags.DEFINE_string('sos', '<sos>', 'start-of-sentence token')
flags.DEFINE_string('eos', '<eos>', 'end-of-sentence token')
flags.DEFINE_integer('unk_id', 0, 'index of unknown token')
flags.DEFINE_integer('hidden_size', 300, 'hidden size of RNN cell')
flags.DEFINE_integer('encoder_num_layers', 2, 'number of layers of encoder')
flags.DEFINE_integer('decoder_num_layers', 2, 'number of layers of decoder')
flags.DEFINE_integer('batch_size', 1, 'batch size')

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

# ======================== SEQ2SEQ NETWORK =============================
def create_network(source_sequence, sos, eos,
                   # target_sequence_in, target_sequence_out,
                   vocab,
                   source_sequence_length,
                   # target_sequence_length,
                   vocab_size,
                   hidden_size, batch_size,
                   encoder_num_layers, decoder_num_layers):
  with tf.variable_scope('encoder'):
    encoder_embedding = tf.get_variable(
      'encoder_embedding_weights',
      [vocab_size, hidden_size],
      dtype=tf.float32,
      initializer=tf.initializers.random_uniform(-1, 1, dtype=tf.float32))

    source_sequence = tf.transpose(source_sequence)
    encoder_embedded = tf.nn.embedding_lookup(encoder_embedding, source_sequence)

    def _create_encoder_cell(hidden_size):
      cell =  tf.nn.rnn_cell.LSTMCell(hidden_size)
      return cell

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

    def _create_decoder_cell(hidden_size):
      cell =  tf.nn.rnn_cell.LSTMCell(hidden_size)
      return cell
    decoder_lstm = tf.nn.rnn_cell.MultiRNNCell(
      [_create_decoder_cell(hidden_size) for _ in range(decoder_num_layers)])
    decoder_lstm = tf.contrib.seq2seq.AttentionWrapper(
      decoder_lstm, attention_mechanism,
      attention_layer_size=hidden_size)
    decoder_output_layer = tf.layers.Dense(vocab_size, use_bias=False)

    decoder_initial_state = decoder_lstm.zero_state(
      batch_size, tf.float32).clone(cell_state=encoder_state)

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

  return preds

int_to_vocab, vocab_size = load_vocab(FLAGS.vocab_file)

vocab = lookup_ops.index_table_from_file(
  FLAGS.vocab_file, default_value=FLAGS.unk_id)
eos_id = tf.cast(vocab.lookup(tf.constant(FLAGS.eos)), tf.int32)

raw_sequence_op = tf.placeholder(tf.string, [1, None])

sequence = tf.cast(vocab.lookup(raw_sequence_op), tf.int32)
sequence = tf.map_fn(lambda x: tf.concat((x, [eos_id]), 0), sequence)
source_sequence_length = tf.map_fn(lambda x: tf.size(x), sequence)

preds = create_network(
  sequence, FLAGS.sos, FLAGS.eos,
  vocab,
  source_sequence_length,
  vocab_size,
  FLAGS.hidden_size, FLAGS.batch_size,
  FLAGS.encoder_num_layers, FLAGS.decoder_num_layers)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

saver = tf.train.Saver()
latest_checkpoint = tf.train.latest_checkpoint('checkpoint_bahdanau')
if latest_checkpoint and tf.train.checkpoint_exists(latest_checkpoint):
  saver.restore(sess, latest_checkpoint)

print('>Chun: ', end='')
raw_sequence = input()
while raw_sequence != 'Shut up!':
  raw_sequence = process_line(raw_sequence)
  raw_sequence = np.array([raw_sequence.split()])
  predictions = sess.run(preds, feed_dict={raw_sequence_op: raw_sequence})
  pred_sent = ' '.join([int_to_vocab[ix] for ix in predictions[:, 0]])
  eos_index = -1
  if FLAGS.eos in pred_sent:
    eos_index = pred_sent.index(FLAGS.eos)
  print('>Bot:', pred_sent[:eos_index])
  print()

  print('>Chun: ', end='')
  raw_sequence = input()
