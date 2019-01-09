import tensorflow as tf
from tensorflow.python.ops import lookup_ops
import codecs
import numpy as np

UNK = '<unk>'
SOS = '<s>'
EOS = '</s>'
UNK_token = 0

src_vocab_size = 1000
encoder_hidden_size = 256
target_vocab_size = 1200
decoder_hidden_size = 300
batch_size = 10
learning_rate = 0.001
source_max_length = 50
target_max_length = 50
max_gradient = 5.0

# ======================== DATA READING =============================
def load_vocab(vocab_file):
  vocab = []
  with codecs.getreader('utf-8')(tf.gfile.GFile(vocab_file, 'r')) as f:
    vocab_size = 0
    for word in f:
      vocab.append(word.strip())
      vocab_size += 1
  return vocab, vocab_size


# From data reader (per batch):
# source sequence [source_max_time, batch_size]
# source sequence length
# target sequence [target_max_time, batch_size]
# target sequence length
def create_input_data(source_data_file, target_data_file,
                      source_vocab_file, target_vocab_file):
  source_dataset = tf.data.TextLineDataset(tf.gfile.Glob(source_data_file))
  target_dataset = tf.data.TextLineDataset(tf.gfile.Glob(target_data_file))
  source_vocab = lookup_ops.index_table_from_file(
    source_vocab_file, default_value=UNK_token)
  target_vocab = lookup_ops.index_table_from_file(
    target_vocab_file, default_value=UNK_token)

  output_buffer_size = batch_size * 1000

  source_eos_id = tf.cast(source_vocab.lookup(tf.constant(EOS)), tf.int32)
  target_sos_id = tf.cast(target_vocab.lookup(tf.constant(SOS)), tf.int32)
  target_eos_id = tf.cast(target_vocab.lookup(tf.constant(EOS)), tf.int32)

  dataset = tf.data.Dataset.zip((source_dataset, target_dataset))
  dataset = dataset.map(
    lambda src, tgt: (tf.string_split([src]).values,
                      tf.string_split([tgt]).values)).prefetch(output_buffer_size)
  dataset = dataset.filter(
    lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))
  dataset = dataset.map(
    lambda src, tgt: (src[:source_max_length], tgt[:target_max_length]))
  dataset = dataset.prefetch(output_buffer_size)

  dataset = dataset.map(
    lambda src, tgt: (tf.cast(source_vocab.lookup(src), tf.int32),
                      tf.cast(target_vocab.lookup(tgt), tf.int32)))
  dataset = dataset.prefetch(output_buffer_size)

  dataset = dataset.map(
    lambda src, tgt: (src,
                      tf.concat(([target_sos_id], tgt), 0),
                      tf.concat((tgt, [target_eos_id]), 0))).prefetch(output_buffer_size)

  dataset = dataset.map(
    lambda src, tgt_in, tgt_out: (
      src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in))).prefetch(output_buffer_size)

  dataset = dataset.repeat().padded_batch(
    batch_size,
    padded_shapes=(tf.TensorShape([None]),
                   tf.TensorShape([None]),
                   tf.TensorShape([None]),
                   tf.TensorShape([]),
                   tf.TensorShape([])),
    padding_values=(source_eos_id,
                   target_eos_id,
                   target_eos_id,
                   0,
                   0))

  iterator = dataset.make_initializable_iterator()

  return iterator.get_next(), iterator.initializer

# ======================== SEQ2SEQ NETWORK =============================
def create_network(source_sequence,
            target_sequence_in, target_sequence_out,
            source_sequence_length,
            target_sequence_length,
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
    encoder_lstm = tf.nn.rnn_cell.LSTMCell(encoder_hidden_size)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
      encoder_lstm,
      encoder_embedded,
      dtype=tf.float32,
      sequence_length=source_sequence_length)

  with tf.variable_scope('decoder'):
    decoder_embedding = tf.get_variable(
      'decoder_embedding_weights',
      [target_vocab_size, decoder_hidden_size],
      dtype=tf.float32,
      initializer=tf.initializers.random_uniform(-1, 1, dtype=tf.float32))
    target_sequence_in = tf.transpose(target_sequence_in)
    decoder_embedded = tf.nn.embedding_lookup(decoder_embedding, target_sequence_in)
    decoder_lstm = tf.nn.rnn_cell.LSTMCell(decoder_hidden_size)
    decoder_output_layer = tf.layers.Dense(target_vocab_size, use_bias=False)

    decoder_initial_state = decoder_lstm.zero_state(batch_size, tf.float32)

    helper = tf.contrib.seq2seq.TrainingHelper(
      decoder_embedded,
      target_sequence_length,
      time_major=True)

    my_decoder = tf.contrib.seq2seq.BasicDecoder(
      decoder_lstm,
      helper,
      decoder_initial_state)

    decoder_outputs, decoder_final_states, _ = tf.contrib.seq2seq.dynamic_decode(
      my_decoder,
      output_time_major=True,
      swap_memory=True)

    logits = decoder_output_layer(decoder_outputs.rnn_output)

    target_sequence_out = tf.transpose(target_sequence_out)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=target_sequence_out, logits=logits)
    loss_weights = tf.sequence_mask(
      target_sequence_length, tf.shape(target_sequence_out)[0],
      dtype=tf.float32)
    loss_weights = tf.transpose(loss_weights)

    loss = tf.reduce_sum(cross_entropy * loss_weights) / tf.to_float(batch_size)
  return loss, source_sequence, target_sequence_in, logits

def create_train_op(loss):
  global_step = tf.Variable(0, trainable=False)

  params = tf.trainable_variables()
  gradients = tf.gradients(
    loss,
    params)
  clipped_gradients, _ = tf.clip_by_global_norm(
    gradients, max_gradient)
  opt = tf.train.AdamOptimizer(learning_rate)
  train_op = opt.apply_gradients(
    zip(clipped_gradients, params), global_step)
  return train_op

source_data_file = '../data/tst2012.vi'
target_data_file = '../data/tst2012.en'
source_vocab_file = '../data/vocab.vi'
target_vocab_file = '../data/vocab.en'

source_vocab, source_vocab_size = load_vocab(source_vocab_file)
target_vocab, target_vocab_size = load_vocab(target_vocab_file)

(source_sequence,
 target_sequence_in, target_sequence_out,
 source_sequence_length, target_sequence_length),\
iterator_initializer = create_input_data(source_data_file, target_data_file,
                                         source_vocab_file, target_vocab_file)

loss, t_source_sequence, t_target_sequence_in, logits = create_network(
  source_sequence,
  target_sequence_in, target_sequence_out,
  source_sequence_length,
  target_sequence_length,
  source_vocab_size,
  target_vocab_size)

train_op = create_train_op(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
sess.run(iterator_initializer)

losses = []
for i in range(10000):
  src_seq, tar_seq, logit_value, loss_value, _ = sess.run(
    [t_source_sequence, t_target_sequence_in, logits, loss, train_op])
  if i % 100 == 0:
    predictions = np.argmax(logit_value, axis=2)
    print('Loss value at step {}: {:.4f}'.format(i + 1, loss_value))
    src_sent = ' '.join([source_vocab[i] for i in src_seq[:, 0]])
    tar_sent = ' '.join([target_vocab[i] for i in tar_seq[:, 0]])
    pred_sent = ' '.join([target_vocab[i] for i in predictions[:, 0]])
    print('<src>', src_sent)
    print('<dst>', tar_sent)
    print('<pred>', pred_sent)
    losses.append(loss_value)

np.savetxt('loss.txt', losses)