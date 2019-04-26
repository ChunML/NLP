# -*- coding: UTF-8 -*-
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import lookup_ops
import codecs
import numpy as np
import sys

# TODO: Use tf.app.flags
UNK = '<unk>'
SOS = '<s>'
EOS = '</s>'
UNK_token = 0
encoder_hidden_size = 512
decoder_hidden_size = 512
encoder_num_layers = 2
decoder_num_layers = 2
src_sent = 'Bạn từ đâu đến ?'

# ======================== DATA READING =============================
def load_vocab(vocab_file):
  vocab = []
  with codecs.getreader('utf-8')(tf.gfile.GFile(vocab_file, 'r')) as f:
    vocab_size = 0
    for word in f:
      vocab.append(word.strip())
      vocab_size += 1
  return vocab, vocab_size

# ======================== SEQ2SEQ NETWORK =============================
def create_network(source_sequence,
                   target_vocab,
                   source_sequence_length,
                   source_vocab_size,
                   target_vocab_size):
  with tf.variable_scope('encoder'):
    encoder_embedding = tf.get_variable(
      'encoder_embedding_weights',
      [source_vocab_size, encoder_hidden_size],
      dtype=tf.float32,
      initializer=tf.initializers.random_uniform(-1, 1, dtype=tf.float32))
    source_sequence = tf.transpose(source_sequence)
    encoder_embedded = tf.nn.embedding_lookup(encoder_embedding, source_sequence)

    # TODO: Update to bidirectional RNN
    def _create_encoder_cell(hidden_size):
      cell =  tf.nn.rnn_cell.LSTMCell(hidden_size)
      return cell
    encoder_lstm = tf.nn.rnn_cell.MultiRNNCell(
      [_create_encoder_cell(encoder_hidden_size) for _ in range(encoder_num_layers)])
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
      encoder_lstm,
      encoder_embedded,
      dtype=tf.float32,
      time_major=True,
      sequence_length=source_sequence_length)

  with tf.variable_scope('decoder'):
    decoder_embedding = tf.get_variable(
      'decoder_embedding_weights',
      [target_vocab_size, decoder_hidden_size],
      dtype=tf.float32,
      initializer=tf.initializers.random_uniform(-1, 1, dtype=tf.float32))

    # TODO: Update to bidirectional RNN
    def _create_decoder_cell(hidden_size):
      cell =  tf.nn.rnn_cell.LSTMCell(hidden_size)
      return cell
    decoder_lstm = tf.nn.rnn_cell.MultiRNNCell(
      [_create_decoder_cell(decoder_hidden_size) for _ in range(decoder_num_layers)])
    decoder_output_layer = tf.layers.Dense(target_vocab_size, use_bias=False)

    decoder_initial_state = encoder_state

    target_sos_id = tf.cast(target_vocab.lookup(tf.constant(SOS)), tf.int32)
    target_eos_id = tf.cast(target_vocab.lookup(tf.constant(EOS)), tf.int32)
    infer_sequence_in = tf.fill([1], target_sos_id)

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

source_vocab_file = '../data/vocab.vi'
target_vocab_file = '../data/vocab.en'

source_int_to_vocab, source_vocab_size = load_vocab(source_vocab_file)
target_int_to_vocab, target_vocab_size = load_vocab(target_vocab_file)

source_vocab = lookup_ops.index_table_from_file(
  source_vocab_file, default_value=UNK_token)
target_vocab = lookup_ops.index_table_from_file(
  target_vocab_file, default_value=UNK_token)

src_sent = tf.convert_to_tensor([src_sent.split()[::-1]])
source_sequence = tf.cast(source_vocab.lookup(src_sent), tf.int32)
source_sequence_length = tf.map_fn(lambda x: tf.size(x), source_sequence)

preds = create_network(
  source_sequence,
  target_vocab,
  source_sequence_length,
  source_vocab_size,
  target_vocab_size)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

saver = tf.train.Saver()
latest_checkpoint = tf.train.latest_checkpoint('checkpoint')
if latest_checkpoint and tf.train.checkpoint_exists(latest_checkpoint):
  saver.restore(sess, latest_checkpoint)
else:
  print('You must train the model first!')
  print('Exiting...')
  sys.exit()

predictions = sess.run(preds)
pred_sent = ' '.join([target_int_to_vocab[ix] for ix in predictions[:, 0]])

if EOS in pred_sent:
  eos_index = pred_sent.index(EOS)
  pred_sent = pred_sent[:eos_index]

print(pred_sent)
